"""
Unit tests for GenericExchangeConnector
"""

import os
import sys
import pytest
import logging
import asyncio
import time
from unittest.mock import patch, MagicMock, AsyncMock, call
import json
import ccxt
from datetime import datetime, timedelta

# Đảm bảo đường dẫn đúng để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup patching trước khi import
# Patch setup_logger trước khi import GenericExchangeConnector
patch('config.logging_config.setup_logger', return_value=logging.getLogger('test_mock')).start()
patch('config.security_config.SecretManager').start()
patch('config.system_config.SystemConfig').start()

# Import các module cần test sau khi đã patch
from data_collectors.exchange_api.generic_connector import GenericExchangeConnector, WebSocketManager

# Logger cho tests
logger = logging.getLogger('test_generic_connector')

# Tạo class test kế thừa từ GenericExchangeConnector để test
class TestExchangeConnector(GenericExchangeConnector):
    """Test implementation của GenericExchangeConnector"""
    
    async def get_exchange_info(self):
        return {"test": "info"}
    
    async def get_trading_fees(self, symbol=None):
        return {"maker": 0.001, "taker": 0.001}
    
    async def subscribe_to_websocket(self, symbol, channels):
        return None


# ========== WEBSOCKET MANAGER TESTS ==========

@pytest.mark.asyncio
async def test_websocket_manager_init():
    """Test khởi tạo WebSocketManager."""
    logger.info("Testing WebSocketManager initialization")
    
    mock_logger = MagicMock()
    ws_manager = WebSocketManager(mock_logger)
    
    assert ws_manager.connections == {}
    assert ws_manager.connection_count == 0
    assert ws_manager.max_connections > 0
    assert ws_manager.symbols_per_connection > 0
    assert ws_manager.processing_callbacks == {}
    
    logger.info("✅ WebSocketManager initialization test passed")


@pytest.mark.asyncio
async def test_websocket_manager_add_connection(patch_websockets):
    """Test thêm kết nối mới vào WebSocketManager."""
    logger.info("Testing WebSocketManager.add_connection")
    
    mock_logger = MagicMock()
    ws_manager = WebSocketManager(mock_logger)
    
    # Mock functions
    connect_func = AsyncMock(return_value=patch_websockets)
    message_callback = AsyncMock()
    
    # Add new connection
    await ws_manager.add_connection(
        "wss://test.com/ws",
        ["BTC/USDT"],
        ["ticker"],
        connect_func,
        message_callback
    )
    
    # Verify connection was added
    assert len(ws_manager.connections) == 1
    assert "conn_0" in ws_manager.connections
    assert ws_manager.connections["conn_0"]["ws_url"] == "wss://test.com/ws"
    assert ws_manager.connections["conn_0"]["symbols"] == ["BTC/USDT"]
    assert ws_manager.connections["conn_0"]["channels"] == ["ticker"]
    assert ws_manager.connections["conn_0"]["task"] is not None
    
    # Verify connect function was called
    connect_func.assert_called_once()
    
    # Clean up
    for conn_info in ws_manager.connections.values():
        conn_info["task"].cancel()
    
    logger.info("✅ WebSocketManager.add_connection test passed")


# ========== GENERIC CONNECTOR TESTS ==========

@pytest.mark.asyncio
async def test_generic_connector_init(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test khởi tạo GenericExchangeConnector."""
    logger.info("Testing GenericExchangeConnector initialization")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    assert connector.exchange_id == 'binance'
    assert connector.api_key == 'test_api_key'
    assert connector.api_secret == 'test_api_secret'
    assert connector.ws_manager is not None
    assert connector.data_buffer is not None
    assert connector.running is True
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector initialization test passed")


@pytest.mark.asyncio
async def test_initialize(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test method initialize."""
    logger.info("Testing GenericExchangeConnector.initialize")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Patch background tasks
    connector._start_background_tasks = MagicMock()
    
    await connector.initialize()
    
    assert connector.markets is not None
    assert connector._start_background_tasks.called
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.initialize test passed")


@pytest.mark.asyncio
async def test_api_retry_decorator(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test API retry decorator."""
    logger.info("Testing api_retry_decorator")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Create a mock function that fails twice then succeeds
    mock_func = AsyncMock(side_effect=[
        ccxt.NetworkError("Test network error"),
        ccxt.ExchangeNotAvailable("Test exchange not available"),
        "success"
    ])
    
    # Apply the decorator
    decorated_func = connector.api_retry_decorator()(mock_func)
    
    # Call the decorated function
    result = await decorated_func()
    
    # Verify the function was called three times
    assert mock_func.call_count == 3
    assert result == "success"
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ api_retry_decorator test passed")


@pytest.mark.asyncio
async def test_fetch_ticker(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test fetch_ticker method."""
    logger.info("Testing GenericExchangeConnector.fetch_ticker")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Test without cache
    ticker = await connector.fetch_ticker("BTC/USDT")
    
    assert ticker is not None
    assert ticker["symbol"] == "BTC/USDT"
    assert "BTC/USDT" in connector.data_buffer["tickers"]
    
    # Test with cache
    cache_ticker = await connector.fetch_ticker("BTC/USDT")
    
    assert cache_ticker == ticker
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.fetch_ticker test passed")


@pytest.mark.asyncio
async def test_fetch_ohlcv(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test fetch_ohlcv method."""
    logger.info("Testing GenericExchangeConnector.fetch_ohlcv")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Test without cache
    candles = await connector.fetch_ohlcv("BTC/USDT", "1h", None, 100)
    
    assert candles is not None
    assert len(candles) > 0
    assert "BTC/USDT_1h_100" in connector.data_buffer["klines"]
    
    # Test with cache
    cache_candles = await connector.fetch_ohlcv("BTC/USDT", "1h", None, 100)
    
    assert cache_candles == candles
    
    # Test with specific since parameter (bypassing cache)
    since = int(time.time() * 1000) - 86400000  # 24 hours ago
    specific_candles = await connector.fetch_ohlcv("BTC/USDT", "1h", since, 100)
    
    assert specific_candles is not None
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.fetch_ohlcv test passed")


@pytest.mark.asyncio
async def test_batch_fetch_ohlcv(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test batch_fetch_ohlcv method."""
    logger.info("Testing GenericExchangeConnector.batch_fetch_ohlcv")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Test batch fetch
    symbols = ["BTC/USDT", "ETH/USDT"]
    results = await connector.batch_fetch_ohlcv(symbols, "1h", None, 100)
    
    assert results is not None
    assert "BTC/USDT" in results
    assert "ETH/USDT" in results
    assert len(results["BTC/USDT"]) > 0
    assert len(results["ETH/USDT"]) > 0
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.batch_fetch_ohlcv test passed")


@pytest.mark.asyncio
async def test_create_order(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test create_order method."""
    logger.info("Testing GenericExchangeConnector.create_order")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock check_connection to return True
    connector.check_connection = AsyncMock(return_value=True)
    
    # Mock markets
    connector.markets = {"BTC/USDT": {}}
    
    # Test create order
    order = await connector.create_order("BTC/USDT", "limit", "buy", 0.1, 50000)
    
    assert order is not None
    assert order["symbol"] == "BTC/USDT"
    assert order["type"] == "limit"
    assert order["side"] == "buy"
    
    # Test create order with non-existent symbol (should reload markets)
    connector.markets = {}
    connector.async_exchange.load_markets = AsyncMock()
    
    with pytest.raises(ValueError):
        await connector.create_order("NON/EXISTENT", "limit", "buy", 0.1, 50000)
    
    # Verify load_markets was called
    assert connector.async_exchange.load_markets.called
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.create_order test passed")


@pytest.mark.asyncio
async def test_cancel_order(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test cancel_order method."""
    logger.info("Testing GenericExchangeConnector.cancel_order")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Test cancel order
    order = await connector.cancel_order("12345", "BTC/USDT")
    
    assert order is not None
    assert order["id"] == "12345"
    
    # Test cancel order with OrderNotFound exception
    connector.async_exchange.cancel_order = AsyncMock(side_effect=ccxt.OrderNotFound("Order not found"))
    connector.fetch_order = AsyncMock(return_value={"id": "12345", "status": "closed"})
    
    order = await connector.cancel_order("12345", "BTC/USDT")
    
    assert order is not None
    assert order["id"] == "12345"
    assert order["status"] == "closed"
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.cancel_order test passed")


@pytest.mark.asyncio
async def test_batch_fetch_tickers(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test batch_fetch_tickers method."""
    logger.info("Testing GenericExchangeConnector.batch_fetch_tickers")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Test batch fetch tickers
    symbols = ["BTC/USDT", "ETH/USDT", "LTC/USDT", "XRP/USDT", "DOT/USDT"]
    results = await connector.batch_fetch_tickers(symbols)
    
    assert results is not None
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.batch_fetch_tickers test passed")


@pytest.mark.asyncio
async def test_historical_data(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test get_historical_data method."""
    logger.info("Testing GenericExchangeConnector.get_historical_data")
    
    connector = TestExchangeConnector(
        exchange_id='binance',
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Patch fetch_ohlcv to return incremental data
    candles_data = []
    for i in range(10):
        timestamp = int(time.time() * 1000) - (86400000 - i * 3600000)  # Starting 24 hours ago
        candle = [timestamp, 50000 + i * 100, 50100 + i * 100, 49900 + i * 100, 50050 + i * 100, 10 + i]
        candles_data.append(candle)
    
    connector.fetch_ohlcv = AsyncMock(return_value=candles_data)
    
    # Test get historical data
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)
    
    historical_data = await connector.get_historical_data(
        "BTC/USDT", "1h", start_time, end_time, 10
    )
    
    assert historical_data is not None
    assert len(historical_data) > 0
    
    # Verify first candle timestamp
    assert historical_data[0][0] >= int(start_time.timestamp() * 1000)
    assert historical_data[-1][0] <= int(end_time.timestamp() * 1000)
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ GenericExchangeConnector.get_historical_data test passed")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])