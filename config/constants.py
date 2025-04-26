"""
Hằng số hệ thống.
File này chứa các hằng số được sử dụng xuyên suốt dự án,
như mã lỗi, trạng thái, và các giá trị mặc định.
"""

from enum import Enum, auto
from typing import Dict, List, Any

# --- Trạng thái hệ thống ---

class SystemStatus(Enum):
    """Trạng thái hệ thống."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"

# --- Trạng thái giao dịch ---

class OrderStatus(Enum):
    """Trạng thái lệnh giao dịch."""
    PENDING = "pending"
    OPEN = "open"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"

class PositionStatus(Enum):
    """Trạng thái vị thế."""
    OPEN = "open"
    CLOSED = "closed"
    LIQUIDATED = "liquidated"

class PositionSide(Enum):
    """Phía vị thế."""
    LONG = "long"
    SHORT = "short"
    BOTH = "both"

class OrderType(Enum):
    """Loại lệnh giao dịch."""
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"

class TimeInForce(Enum):
    """Hiệu lực thời gian của lệnh."""
    GTC = "gtc"  # Good Till Cancel
    IOC = "ioc"  # Immediate Or Cancel
    FOK = "fok"  # Fill Or Kill
    GTX = "gtx"  # Good Till Crossing

# --- Mã lỗi ---

class ErrorCode(Enum):
    """Mã lỗi hệ thống."""
    # Lỗi chung
    UNKNOWN_ERROR = 1000
    INVALID_PARAMETER = 1001
    CONFIGURATION_ERROR = 1002
    PERMISSION_DENIED = 1003
    TIMEOUT_ERROR = 1004
    
    # Lỗi kết nối
    CONNECTION_ERROR = 2000
    API_ERROR = 2001
    RATE_LIMIT_EXCEEDED = 2002
    AUTHENTICATION_FAILED = 2003
    
    # Lỗi dữ liệu
    DATA_NOT_FOUND = 3000
    DATA_CORRUPTED = 3001
    INVALID_DATA_FORMAT = 3002
    MISSING_REQUIRED_DATA = 3003
    
    # Lỗi giao dịch
    INSUFFICIENT_BALANCE = 4000
    ORDER_REJECTED = 4001
    POSITION_NOT_FOUND = 4002
    INVALID_ORDER_PARAMS = 4003
    MARKET_CLOSED = 4004
    
    # Lỗi huấn luyện
    TRAINING_FAILED = 5000
    MODEL_NOT_FOUND = 5001
    INVALID_MODEL_STATE = 5002
    ENVIRONMENT_ERROR = 5003
    
    # Lỗi hệ thống
    SYSTEM_OVERLOAD = 6000
    CRITICAL_ERROR = 6001
    SHUTDOWN_REQUIRED = 6002

# --- Thông báo mặc định ---

ERROR_MESSAGES = {
    ErrorCode.UNKNOWN_ERROR: "Đã xảy ra lỗi không xác định",
    ErrorCode.INVALID_PARAMETER: "Tham số không hợp lệ",
    ErrorCode.CONFIGURATION_ERROR: "Lỗi cấu hình hệ thống",
    ErrorCode.PERMISSION_DENIED: "Không có quyền thực hiện hành động này",
    ErrorCode.TIMEOUT_ERROR: "Quá thời gian chờ",
    
    ErrorCode.CONNECTION_ERROR: "Lỗi kết nối",
    ErrorCode.API_ERROR: "Lỗi API",
    ErrorCode.RATE_LIMIT_EXCEEDED: "Vượt quá giới hạn tần suất yêu cầu",
    ErrorCode.AUTHENTICATION_FAILED: "Xác thực thất bại",
    
    ErrorCode.DATA_NOT_FOUND: "Không tìm thấy dữ liệu",
    ErrorCode.DATA_CORRUPTED: "Dữ liệu bị hỏng",
    ErrorCode.INVALID_DATA_FORMAT: "Định dạng dữ liệu không hợp lệ",
    ErrorCode.MISSING_REQUIRED_DATA: "Thiếu dữ liệu bắt buộc",
    
    ErrorCode.INSUFFICIENT_BALANCE: "Số dư không đủ",
    ErrorCode.ORDER_REJECTED: "Lệnh bị từ chối",
    ErrorCode.POSITION_NOT_FOUND: "Không tìm thấy vị thế",
    ErrorCode.INVALID_ORDER_PARAMS: "Tham số lệnh không hợp lệ",
    ErrorCode.MARKET_CLOSED: "Thị trường đã đóng cửa",
    
    ErrorCode.TRAINING_FAILED: "Huấn luyện thất bại",
    ErrorCode.MODEL_NOT_FOUND: "Không tìm thấy mô hình",
    ErrorCode.INVALID_MODEL_STATE: "Trạng thái mô hình không hợp lệ",
    ErrorCode.ENVIRONMENT_ERROR: "Lỗi môi trường huấn luyện",
    
    ErrorCode.SYSTEM_OVERLOAD: "Hệ thống quá tải",
    ErrorCode.CRITICAL_ERROR: "Lỗi nghiêm trọng",
    ErrorCode.SHUTDOWN_REQUIRED: "Yêu cầu tắt hệ thống",
}

# --- Danh sách sàn giao dịch hỗ trợ ---

class Exchange(Enum):
    """Danh sách sàn giao dịch hỗ trợ."""
    BINANCE = "binance"
    BYBIT = "bybit"
    BITMEX = "bitmex"
    OKEX = "okex"
    HUOBI = "huobi"
    FTX = "ftx"
    COINBASE = "coinbase"
    KRAKEN = "kraken"

# Các sàn có API hoạt động (cần cập nhật nếu có thay đổi)
ACTIVE_EXCHANGES = [
    Exchange.BINANCE,
    Exchange.BYBIT,
    Exchange.OKEX,
    Exchange.COINBASE,
    Exchange.KRAKEN
]

# --- Timeframes ---

class Timeframe(Enum):
    """Khung thời gian giao dịch."""
    M1 = "1m"   # 1 phút
    M5 = "5m"   # 5 phút
    M15 = "15m" # 15 phút
    M30 = "30m" # 30 phút
    H1 = "1h"   # 1 giờ
    H4 = "4h"   # 4 giờ
    D1 = "1d"   # 1 ngày
    W1 = "1w"   # 1 tuần

# Bảng chuyển đổi timeframe sang second
TIMEFRAME_TO_SECONDS = {
    Timeframe.M1.value: 60,
    Timeframe.M5.value: 300,
    Timeframe.M15.value: 900,
    Timeframe.M30.value: 1800,
    Timeframe.H1.value: 3600,
    Timeframe.H4.value: 14400,
    Timeframe.D1.value: 86400,
    Timeframe.W1.value: 604800,
}

# Danh sách timeframe được hỗ trợ cho mỗi sàn
EXCHANGE_TIMEFRAMES = {
    Exchange.BINANCE.value: [
        Timeframe.M1.value, Timeframe.M5.value, Timeframe.M15.value, 
        Timeframe.M30.value, Timeframe.H1.value, Timeframe.H4.value, 
        Timeframe.D1.value, Timeframe.W1.value
    ],
    Exchange.BYBIT.value: [
        Timeframe.M1.value, Timeframe.M5.value, Timeframe.M15.value, 
        Timeframe.M30.value, Timeframe.H1.value, Timeframe.H4.value, 
        Timeframe.D1.value, Timeframe.W1.value
    ],
    # Thêm các sàn khác nếu cần
}

# --- Chỉ báo kỹ thuật ---

class Indicator(Enum):
    """Chỉ báo kỹ thuật."""
    # Trend indicators
    SMA = "sma"         # Simple Moving Average
    EMA = "ema"         # Exponential Moving Average
    MACD = "macd"       # Moving Average Convergence Divergence
    BOLLINGER = "bb"    # Bollinger Bands
    
    # Momentum indicators
    RSI = "rsi"         # Relative Strength Index
    STOCH = "stoch"     # Stochastic Oscillator
    CCI = "cci"         # Commodity Channel Index
    
    # Volume indicators
    OBV = "obv"         # On-Balance Volume
    ADI = "adi"         # Accumulation/Distribution Index
    
    # Volatility indicators
    ATR = "atr"         # Average True Range
    
    # Support/Resistance
    SR_LEVELS = "sr"    # Support/Resistance Levels
    PIVOT = "pivot"     # Pivot Points

# Tham số mặc định cho các chỉ báo
DEFAULT_INDICATOR_PARAMS = {
    Indicator.SMA.value: {"window": 20},
    Indicator.EMA.value: {"window": 20, "alpha": 0.1},
    Indicator.MACD.value: {"fast": 12, "slow": 26, "signal": 9},
    Indicator.BOLLINGER.value: {"window": 20, "stds": 2},
    Indicator.RSI.value: {"window": 14},
    Indicator.STOCH.value: {"k": 14, "d": 3, "smooth_k": 3},
    Indicator.CCI.value: {"window": 20},
    Indicator.OBV.value: {},
    Indicator.ADI.value: {},
    Indicator.ATR.value: {"window": 14},
    Indicator.SR_LEVELS.value: {"window": 200, "min_touches": 2},
    Indicator.PIVOT.value: {"type": "standard"},
}

# --- Tham số huấn luyện ---

class AgentType(Enum):
    """Loại agent."""
    DQN = "dqn"
    PPO = "ppo"
    A2C = "a2c"
    DDPG = "ddpg"
    SAC = "sac"
    TD3 = "td3"

# Tham số mặc định cho các loại agent
DEFAULT_AGENT_PARAMS = {
    AgentType.DQN.value: {
        "gamma": 0.99,
        "learning_rate": 0.001,
        "batch_size": 64,
        "buffer_size": 10000,
        "update_target_every": 100,
        "hidden_layers": [64, 64],
    },
    AgentType.PPO.value: {
        "gamma": 0.99,
        "learning_rate": 0.0003,
        "clip_ratio": 0.2,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "lam": 0.95,
        "batch_size": 64,
        "epochs": 10,
        "hidden_layers": [64, 64],
    },
    AgentType.A2C.value: {
        "gamma": 0.99,
        "learning_rate": 0.0007,
        "value_coef": 0.5,
        "entropy_coef": 0.01,
        "rms_prop_eps": 1e-5,
        "hidden_layers": [64, 64],
    },
    # Thêm các agent khác nếu cần
}

class RewardFunction(Enum):
    """Loại hàm phần thưởng."""
    PROFIT = "profit"
    RISK_ADJUSTED = "risk_adjusted"
    SHARPE = "sharpe"
    SORTINO = "sortino"
    CALMAR = "calmar"
    CUSTOM = "custom"

# --- Tham số backtest ---

class BacktestMetric(Enum):
    """Các chỉ số đánh giá backtest."""
    TOTAL_RETURN = "total_return"
    ANNUALIZED_RETURN = "annualized_return"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    MAX_DRAWDOWN = "max_drawdown"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"
    PROFIT_FACTOR = "profit_factor"
    EXPECTANCY = "expectancy"
    AVERAGE_TRADE = "average_trade"
    AVERAGE_WIN = "average_win"
    AVERAGE_LOSS = "average_loss"
    RECOVERY_FACTOR = "recovery_factor"
    RISK_REWARD_RATIO = "risk_reward_ratio"

# --- Hằng số khác ---

# Tỷ lệ học tối đa và tối thiểu
MIN_LEARNING_RATE = 1e-6
MAX_LEARNING_RATE = 1e-2

# Giới hạn độ lớn batch trong huấn luyện
MIN_BATCH_SIZE = 16
MAX_BATCH_SIZE = 1024

# Số lượng mẫu tối thiểu cho huấn luyện
MIN_SAMPLES_FOR_TRAINING = 1000

# Giới hạn đòn bẩy
MIN_LEVERAGE = 1.0
MAX_LEVERAGE = 100.0

# Giới hạn rủi ro trên mỗi giao dịch
MAX_RISK_PER_TRADE = 0.05  # 5% số dư

# Số lượng vị thế mở tối đa
MAX_OPEN_POSITIONS = 10

# Số lượng lần thử lại tối đa cho các request API
MAX_API_RETRIES = 3

# Timeout cho các request API (giây)
API_TIMEOUT = 30

# Số lượng điểm dữ liệu tối đa để lưu trong bộ nhớ
MAX_DATA_POINTS_IN_MEMORY = 10000

# Độ trễ tối đa cho phép (ms)
MAX_LATENCY_MS = 200

# --- Đường dẫn file mặc định ---

# Đường dẫn mặc định cho data
DEFAULT_DATA_PATHS = {
    "historical": "data/historical",
    "market": "data/market",
    "orderbook": "data/orderbook",
    "news": "data/news",
    "sentiment": "data/sentiment",
}

# Đường dẫn mặc định cho model
DEFAULT_MODEL_PATHS = {
    "checkpoints": "saved_models/checkpoints",
    "best_models": "saved_models/best_models",
    "hyperparams": "saved_models/hyperparams",
}