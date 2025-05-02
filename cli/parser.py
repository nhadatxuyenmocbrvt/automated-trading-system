"""
Parser dòng lệnh cho hệ thống giao dịch tự động.
File này định nghĩa parser chính và các subparser cho từng lệnh.
"""

import argparse
from datetime import datetime

def create_parser():
    """
    Tạo parser lệnh chính và các subparser.
    
    Returns:
        Parser lệnh đã được cấu hình
    """
    # Parser chính
    parser = argparse.ArgumentParser(
        description="Hệ thống giao dịch tự động với Reinforcement Learning",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Các tham số chung
    parser.add_argument(
        "--mode",
        choices=["development", "testing", "production"],
        default="development",
        help="Chế độ hoạt động của hệ thống"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Hiển thị thông tin chi tiết"
    )
    parser.add_argument(
        "--save-state",
        action="store_true",
        help="Lưu trạng thái hệ thống sau khi thực hiện lệnh"
    )
    
    # Tạo các subparser
    subparsers = parser.add_subparsers(
        dest="command",
        title="Các lệnh",
        help="Lệnh cần thực hiện"
    )
    
    # Subparser cho lệnh collect
    collect_parser = subparsers.add_parser(
        "collect",
        help="Thu thập dữ liệu từ sàn giao dịch"
    )
    _add_collect_parser_arguments(collect_parser)
    
    # Subparser cho lệnh process
    process_parser = subparsers.add_parser(
        "process",
        help="Xử lý dữ liệu thị trường"
    )
    _add_process_parser_arguments(process_parser)
    
    # Subparser cho lệnh backtest
    backtest_parser = subparsers.add_parser(
        "backtest",
        help="Backtest chiến lược giao dịch"
    )
    _add_backtest_parser_arguments(backtest_parser)
    
    # Subparser cho lệnh train
    train_parser = subparsers.add_parser(
        "train",
        help="Huấn luyện agent giao dịch"
    )
    _add_train_parser_arguments(train_parser)
    
    # Subparser cho lệnh trade
    trade_parser = subparsers.add_parser(
        "trade",
        help="Giao dịch thực tế với agent đã huấn luyện"
    )
    _add_trade_parser_arguments(trade_parser)
    
    # Subparser cho lệnh dashboard
    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Khởi chạy dashboard trực quan"
    )
    _add_dashboard_parser_arguments(dashboard_parser)
    
    return parser

def _add_collect_parser_arguments(parser):
    """
    Thêm tham số cho lệnh collect.
    
    Args:
        parser: Parser cần thêm tham số
    """
    parser.add_argument(
        "--exchange",
        required=True,
        help="Sàn giao dịch (binance, bybit, ...)"
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Các cặp giao dịch (BTC/USDT, ETH/USDT, ...)"
    )
    parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h"],
        help="Các khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)"
    )
    
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument(
        "--days",
        type=int,
        help="Số ngày cần lấy dữ liệu (tính từ hiện tại)"
    )
    time_group.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Ngày bắt đầu lấy dữ liệu (định dạng: YYYY-MM-DD)"
    )
    
    parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Ngày kết thúc lấy dữ liệu (định dạng: YYYY-MM-DD, mặc định: hiện tại)"
    )
    parser.add_argument(
        "--futures",
        action="store_true",
        help="Thu thập dữ liệu từ thị trường futures thay vì spot"
    )
    parser.add_argument(
        "--output-dir",
        help="Thư mục đầu ra (mặc định: data/raw)"
    )
    parser.add_argument(
        "--api-key",
        help="API key của sàn giao dịch"
    )
    parser.add_argument(
        "--api-secret",
        help="API secret của sàn giao dịch"
    )

def _add_process_parser_arguments(parser):
    """
    Thêm tham số cho lệnh process.
    
    Args:
        parser: Parser cần thêm tham số
    """
    subparsers = parser.add_subparsers(
        dest="process_command",
        title="Lệnh xử lý",
        help="Loại xử lý cần thực hiện"
    )
    
    # Lệnh con clean - làm sạch dữ liệu
    clean_parser = subparsers.add_parser(
        "clean",
        help="Làm sạch dữ liệu"
    )
    clean_parser.add_argument(
        "--data-type",
        choices=["ohlcv", "orderbook", "trades", "sentiment"],
        default="ohlcv",
        help="Loại dữ liệu cần làm sạch"
    )
    clean_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Các cặp giao dịch cần xử lý"
    )
    clean_parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Các khung thời gian cần xử lý"
    )
    clean_parser.add_argument(
        "--input-dir",
        help="Thư mục đầu vào (mặc định: data/raw)"
    )
    clean_parser.add_argument(
        "--output-dir",
        help="Thư mục đầu ra (mặc định: data/cleaned)"
    )
    
    # Lệnh con features - tạo đặc trưng
    features_parser = subparsers.add_parser(
        "features",
        help="Tạo đặc trưng kỹ thuật"
    )
    features_parser.add_argument(
        "--data-type",
        choices=["ohlcv", "orderbook", "trades", "sentiment"],
        default="ohlcv",
        help="Loại dữ liệu cần tạo đặc trưng"
    )
    features_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Các cặp giao dịch cần xử lý"
    )
    features_parser.add_argument(
        "--timeframes",
        nargs="+",
        help="Các khung thời gian cần xử lý"
    )
    features_parser.add_argument(
        "--indicators",
        nargs="+",
        help="Danh sách chỉ báo kỹ thuật cần tạo"
    )
    features_parser.add_argument(
        "--all-indicators",
        action="store_true",
        help="Tạo tất cả các chỉ báo kỹ thuật có sẵn"
    )
    features_parser.add_argument(
        "--input-dir",
        help="Thư mục đầu vào (mặc định: data/cleaned)"
    )
    features_parser.add_argument(
        "--output-dir",
        help="Thư mục đầu ra (mặc định: data/features)"
    )
    
    # Lệnh con pipeline - chạy toàn bộ pipeline xử lý
    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Chạy toàn bộ pipeline xử lý dữ liệu"
    )
    pipeline_parser.add_argument(
        "--exchange",
        help="Sàn giao dịch (binance, bybit, ...)"
    )
    pipeline_parser.add_argument(
        "--symbols",
        nargs="+",
        help="Các cặp giao dịch cần xử lý"
    )
    pipeline_parser.add_argument(
        "--timeframes",
        nargs="+",
        default=["1h"],
        help="Các khung thời gian cần xử lý"
    )
    pipeline_parser.add_argument(
        "--start-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Ngày bắt đầu lấy dữ liệu (định dạng: YYYY-MM-DD)"
    )
    pipeline_parser.add_argument(
        "--end-date",
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
        help="Ngày kết thúc lấy dữ liệu (định dạng: YYYY-MM-DD, mặc định: hiện tại)"
    )
    pipeline_parser.add_argument(
        "--input-files",
        nargs="+",
        help="Danh sách file đầu vào thay vì thu thập dữ liệu mới"
    )
    pipeline_parser.add_argument(
        "--skip-clean",
        action="store_true",
        help="Bỏ qua bước làm sạch dữ liệu"
    )
    pipeline_parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Bỏ qua bước tạo đặc trưng"
    )
    pipeline_parser.add_argument(
        "--output-dir",
        help="Thư mục đầu ra (mặc định: data/processed)"
    )
    pipeline_parser.add_argument(
        "--pipeline-name",
        help="Tên pipeline xử lý (nếu đã đăng ký trước đó)"
    )

def _add_backtest_parser_arguments(parser):
    """
    Thêm tham số cho lệnh backtest.
    
    Args:
        parser: Parser cần thêm tham số
    """
    parser.add_argument(
        "--agent",
        choices=["dqn", "ppo", "a2c"],
        default="dqn",
        help="Loại agent sử dụng cho backtest"
    )
    parser.add_argument(
        "--model-path",
        help="Đường dẫn đến file mô hình agent đã huấn luyện"
    )
    parser.add_argument(
        "--data-dir",
        help="Thư mục chứa dữ liệu đã xử lý"
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Cặp giao dịch sử dụng cho backtest"
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Khung thời gian sử dụng cho backtest"
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Số dư ban đầu"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Số vị thế tối đa"
    )
    parser.add_argument(
        "--commission",
        type=float,
        default=0.001,
        help="Tỷ lệ phí giao dịch (0.001 = 0.1%)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        help="Tỷ lệ dừng lỗ (0.05 = 5%)"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        help="Tỷ lệ chốt lời (0.1 = 10%)"
    )
    parser.add_argument(
        "--render-mode",
        choices=["console", "human", "rgb_array", "none"],
        default="console",
        help="Chế độ hiển thị backtest"
    )
    parser.add_argument(
        "--output-path",
        help="Đường dẫn lưu kết quả backtest"
    )

def _add_train_parser_arguments(parser):
    """
    Thêm tham số cho lệnh train.
    
    Args:
        parser: Parser cần thêm tham số
    """
    parser.add_argument(
        "--agent",
        choices=["dqn", "ppo", "a2c"],
        default="dqn",
        help="Loại agent cần huấn luyện"
    )
    parser.add_argument(
        "--data-dir",
        help="Thư mục chứa dữ liệu đã xử lý"
    )
    parser.add_argument(
        "--symbol",
        default="BTC/USDT",
        help="Cặp giao dịch sử dụng cho huấn luyện"
    )
    parser.add_argument(
        "--timeframe",
        default="1h",
        help="Khung thời gian sử dụng cho huấn luyện"
    )
    parser.add_argument(
        "--initial-balance",
        type=float,
        default=10000.0,
        help="Số dư ban đầu cho môi trường huấn luyện"
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=5,
        help="Số vị thế tối đa"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Kích thước cửa sổ dữ liệu"
    )
    parser.add_argument(
        "--reward-function",
        choices=["profit", "risk_adjusted", "sharpe", "sortino", "calmar", "custom"],
        default="profit",
        help="Hàm phần thưởng sử dụng"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=1000,
        help="Số episode huấn luyện"
    )
    parser.add_argument(
        "--eval-interval",
        type=int,
        default=100,
        help="Khoảng thời gian đánh giá (số episode)"
    )
    parser.add_argument(
        "--save-interval",
        type=int,
        default=100,
        help="Khoảng thời gian lưu mô hình (số episode)"
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        help="Tải mô hình có sẵn thay vì khởi tạo mới"
    )
    parser.add_argument(
        "--model-path",
        help="Đường dẫn đến file mô hình (khi load-model=True)"
    )
    parser.add_argument(
        "--save-path",
        help="Đường dẫn lưu mô hình mới"
    )
    parser.add_argument(
        "--tensorboard",
        action="store_true",
        help="Ghi log TensorBoard trong quá trình huấn luyện"
    )
    parser.add_argument(
        "--render-training",
        action="store_true",
        help="Hiển thị trực quan trong quá trình huấn luyện"
    )

def _add_trade_parser_arguments(parser):
    """
    Thêm tham số cho lệnh trade.
    
    Args:
        parser: Parser cần thêm tham số
    """
    parser.add_argument(
        "--exchange",
        required=True,
        help="Sàn giao dịch (binance, bybit, ...)"
    )
    parser.add_argument(
        "--symbol",
        required=True,
        help="Cặp giao dịch"
    )
    parser.add_argument(
        "--agent",
        choices=["dqn", "ppo", "a2c"],
        default="dqn",
        help="Loại agent sử dụng"
    )
    parser.add_argument(
        "--model-path",
        required=True,
        help="Đường dẫn đến file mô hình agent đã huấn luyện"
    )
    parser.add_argument(
        "--api-key",
        help="API key của sàn giao dịch"
    )
    parser.add_argument(
        "--api-secret",
        help="API secret của sàn giao dịch"
    )
    parser.add_argument(
        "--futures",
        action="store_true",
        help="Giao dịch trên thị trường futures thay vì spot"
    )
    parser.add_argument(
        "--test-mode",
        action="store_true",
        help="Chạy ở chế độ test (không thực hiện giao dịch thực tế)"
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        help="Số giao dịch tối đa trước khi dừng"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Thời gian tối đa chạy trước khi dừng (giây)"
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        help="Tỷ lệ dừng lỗ (0.05 = 5%)"
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        help="Tỷ lệ chốt lời (0.1 = 10%)"
    )
    parser.add_argument(
        "--max-position-size",
        type=float,
        help="Kích thước vị thế tối đa (% số dư)"
    )

def _add_dashboard_parser_arguments(parser):
    """
    Thêm tham số cho lệnh dashboard.
    
    Args:
        parser: Parser cần thêm tham số
    """
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Cổng chạy dashboard Streamlit"
    )
    parser.add_argument(
        "--data-dir",
        help="Thư mục chứa dữ liệu đã xử lý"
    )
    parser.add_argument(
        "--model-dir",
        help="Thư mục chứa các mô hình đã lưu"
    )
    parser.add_argument(
        "--log-dir",
        help="Thư mục chứa các file log"
    )
    parser.add_argument(
        "--config-file",
        help="File cấu hình hệ thống"
    )