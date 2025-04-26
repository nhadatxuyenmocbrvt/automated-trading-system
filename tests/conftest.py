"""
Pytest configuration file với các fixtures chung để sử dụng trong các test cases.
"""

import os
import json
import pytest
import logging
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import ccxt
import ccxt.async_support as ccxt_async

# Thêm logging cho tests
log_format = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
logging.basicConfig(
    level=logging.DEBUG,
    format=log_format,
    handlers=[
        logging.FileHandler('tests/test_exchange_connectors.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('test_exchanges')

# Giá trị mặc định cho các test
DEFAULT_TEST_SYMBOL = "BTC/USDT"
DEFAULT_TEST_TIMEFRAME = "1h"
DEFAULT_TEST_EXCHANGE_ID = "binance"

# Patching các module config
@pytest.fixture(autouse=True)
def patch_modules():
    """Patch các module cần thiết cho tất cả tests."""
    # Patch config modules
    with patch('config.logging_config.setup_logger', return_value=logging.getLogger('test_mock')), \
         patch('config.logging_config.get_logger', return_value=logging.getLogger('test_mock')), \
         patch('config.security_config.SecretManager'), \
         patch('config.system_config.SystemConfig'):
        yield


# ========== CONFIGURATION MOCKS ==========

@pytest.fixture
def mock_system_config():
    """Mock cho SystemConfig."""
    config = MagicMock()
    config.get_exchange_config.return_value = {
        'api_key': 'test_api_key',
        'api_secret': 'test_api_secret',
        'api_passphrase': None,
        'sandbox': True,
        'rate_limit': True
    }
    config.get_retry_settings.return_value = {
        'max_retries': 3,
        'retry_delay': 1,
        'max_delay': 30
    }
    return config


@pytest.fixture
def mock_secret_manager():
    """Mock cho SecretManager."""
    secret_mgr = MagicMock()
    secret_mgr.get_secret.return_value = 'test_secret'
    secret_mgr.encrypt.return_value = 'encrypted_data'
    secret_mgr.decrypt.return_value = 'decrypted_data'
    return secret_mgr


# ========== CCXT EXCHANGE MOCKS ==========

@pytest.fixture
def mock_exchange_data():
    """Mock data để sử dụng trong các tests."""
    with open('tests/fixtures/mock_exchange_data.json', 'r') as f:
        return json.load(f)


@pytest.fixture
def mock_ccxt_exchange(mock_exchange_data):
    """Mock cho ccxt exchange."""
    exchange = MagicMock()
    
    # Mock market data
    exchange.load_markets.return_value = mock_exchange_data['markets']
    exchange.fetch_ticker.return_value = mock_exchange_data['ticker']
    exchange.fetch_order_book.return_value = mock_exchange_data['orderbook']
    exchange.fetch_ohlcv.return_value = mock_exchange_data['ohlcv']
    exchange.fetch_trades.return_value = mock_exchange_data['trades']
    exchange.fetch_time.return_value = mock_exchange_data['server_time']
    
    # Mock order functions
    exchange.create_order.return_value = mock_exchange_data['order']
    exchange.cancel_order.return_value = mock_exchange_data['order']
    exchange.fetch_order.return_value = mock_exchange_data['order']
    exchange.fetch_open_orders.return_value = [mock_exchange_data['order']]
    exchange.fetch_balance.return_value = mock_exchange_data['balance']
    exchange.fetch_positions.return_value = mock_exchange_data['positions']
    
    # Mock other properties
    exchange.rateLimit = 100
    exchange.sandbox = True
    exchange.id = DEFAULT_TEST_EXCHANGE_ID
    
    return exchange


@pytest.fixture
def mock_ccxt_async_exchange(mock_exchange_data):
    """Mock cho ccxt.async_support exchange."""
    exchange = AsyncMock()
    
    # Mock market data
    exchange.load_markets = AsyncMock(return_value=mock_exchange_data['markets'])
    exchange.fetch_ticker = AsyncMock(return_value=mock_exchange_data['ticker'])
    exchange.fetch_order_book = AsyncMock(return_value=mock_exchange_data['orderbook'])
    exchange.fetch_ohlcv = AsyncMock(return_value=mock_exchange_data['ohlcv'])
    exchange.fetch_trades = AsyncMock(return_value=mock_exchange_data['trades'])
    exchange.fetch_time = AsyncMock(return_value=mock_exchange_data['server_time'])
    
    # Mock order functions
    exchange.create_order = AsyncMock(return_value=mock_exchange_data['order'])
    exchange.cancel_order = AsyncMock(return_value=mock_exchange_data['order'])
    exchange.fetch_order = AsyncMock(return_value=mock_exchange_data['order'])
    exchange.fetch_open_orders = AsyncMock(return_value=[mock_exchange_data['order']])
    exchange.fetch_balance = AsyncMock(return_value=mock_exchange_data['balance'])
    exchange.fetch_positions = AsyncMock(return_value=mock_exchange_data['positions'])
    exchange.close = AsyncMock()
    
    # Mock other properties
    exchange.rateLimit = 100
    exchange.sandbox = True
    exchange.id = DEFAULT_TEST_EXCHANGE_ID
    
    return exchange


@pytest.fixture
def mock_websocket():
    """Mock cho websocket."""
    ws = AsyncMock()
    ws.recv = AsyncMock(side_effect=[
        json.dumps({"event": "subscribed", "channel": "ticker"}),
        json.dumps({"data": {"symbol": "BTCUSDT", "price": "50000"}}),
        json.dumps({"data": {"symbol": "ETHUSDT", "price": "3000"}}),
        asyncio.CancelledError()  # Để kết thúc vòng lặp
    ])
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    return ws


# ========== PATCH FIXTURES ==========

@pytest.fixture
def patch_ccxt_exchanges(mock_ccxt_exchange, mock_ccxt_async_exchange):
    """Patch ccxt và ccxt.async_support để sử dụng mock."""
    
    def mock_getattr_ccxt(obj, name):
        mock = MagicMock()
        mock.return_value = mock_ccxt_exchange
        return mock
    
    def mock_getattr_ccxt_async(obj, name):
        mock = AsyncMock()
        mock.return_value = mock_ccxt_async_exchange
        return mock
    
    with patch('ccxt.__getattr__', mock_getattr_ccxt), \
         patch('ccxt.async_support.__getattr__', mock_getattr_ccxt_async):
        yield


@pytest.fixture
def patch_websockets(mock_websocket):
    """Patch websockets.connect để sử dụng mock."""
    
    async def mock_connect(*args, **kwargs):
        return mock_websocket
    
    with patch('websockets.connect', side_effect=mock_connect):
        yield mock_websocket


# ========== HELPER FIXTURES ==========

@pytest.fixture
def event_loop():
    """Create an instance of the default event loop for each test case."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def create_mock_fixtures_file():
    """Tạo file mock_exchange_data.json nếu chưa tồn tại."""
    os.makedirs('tests/fixtures', exist_ok=True)
    
    if not os.path.exists('tests/fixtures/mock_exchange_data.json'):
        mock_data = {
            "markets": {
                "BTC/USDT": {
                    "id": "BTCUSDT",
                    "symbol": "BTC/USDT",
                    "base": "BTC",
                    "quote": "USDT",
                    "active": True,
                    "precision": {"price": 2, "amount": 8},
                    "limits": {"amount": {"min": 0.0001}, "price": {"min": 0.01}}
                },
                "ETH/USDT": {
                    "id": "ETHUSDT",
                    "symbol": "ETH/USDT",
                    "base": "ETH",
                    "quote": "USDT",
                    "active": True,
                    "precision": {"price": 2, "amount": 8},
                    "limits": {"amount": {"min": 0.001}, "price": {"min": 0.01}}
                }
            },
            "ticker": {
                "symbol": "BTC/USDT",
                "timestamp": 1619712000000,
                "datetime": "2021-04-29T16:00:00.000Z",
                "high": 55000,
                "low": 52000,
                "bid": 53500,
                "ask": 53550,
                "last": 53525,
                "close": 53525,
                "baseVolume": 1000,
                "quoteVolume": 53525000
            },
            "orderbook": {
                "symbol": "BTC/USDT",
                "timestamp": 1619712000000,
                "datetime": "2021-04-29T16:00:00.000Z",
                "bids": [[53500, 1.5], [53450, 2.0]],
                "asks": [[53550, 1.0], [53600, 1.5]]
            },
            "ohlcv": [
                [1619712000000, 53000, 55000, 52000, 53525, 1000],
                [1619715600000, 53525, 54000, 53000, 53750, 800]
            ],
            "trades": [
                {
                    "id": "12345",
                    "timestamp": 1619712000000,
                    "datetime": "2021-04-29T16:00:00.000Z",
                    "symbol": "BTC/USDT",
                    "side": "buy",
                    "price": 53500,
                    "amount": 0.1
                }
            ],
            "order": {
                "id": "12345",
                "timestamp": 1619712000000,
                "datetime": "2021-04-29T16:00:00.000Z",
                "symbol": "BTC/USDT",
                "type": "limit",
                "side": "buy",
                "price": 53000,
                "amount": 0.1,
                "status": "open"
            },
            "balance": {
                "BTC": {"free": 1.0, "used": 0.1, "total": 1.1},
                "USDT": {"free": 50000, "used": 5350, "total": 55350}
            },
            "positions": [
                {
                    "symbol": "BTC/USDT",
                    "side": "long",
                    "amount": 0.1,
                    "entryPrice": 53000,
                    "leverage": 10
                }
            ],
            "server_time": 1619712000000
        }
        
        with open('tests/fixtures/mock_exchange_data.json', 'w') as f:
            json.dump(mock_data, f, indent=4)
        
        logger.info("Created mock_exchange_data.json file")