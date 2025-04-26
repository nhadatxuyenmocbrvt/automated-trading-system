"""
Unit tests for Binance Connector.
"""

import os
import sys
import unittest
import asyncio
from unittest.mock import patch, MagicMock
import json

# Thêm thư mục gốc vào đường dẫn để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collectors.exchange_api.binance_connector import BinanceConnector
from config.system_config import SystemConfig
from config.security_config import SecretManager


class TestBinanceConnector(unittest.TestCase):
    """Test cases for BinanceConnector class."""

    def setUp(self):
        """Set up test environment before each test."""
        self.config = SystemConfig()
        self.secret_manager = SecretManager()
        
        # Dữ liệu mẫu để test
        self.sample_ticker = {
            'symbol': 'BTC/USDT', 
            'timestamp': 1618123456789,
            'datetime': '2021-04-11T12:34:56.789Z',
            'high': 60000, 
            'low': 55000,
            'bid': 58000, 
            'ask': 58100,
            'last': 58050,
            'close': 58050,
            'open': 57000,
            'volume': 1000
        }
        
        self.sample_orderbook = {
            'symbol': 'BTC/USDT',
            'bids': [[58000, 1.5], [57900, 2.0]],
            'asks': [[58100, 1.0], [58200, 2.5]],
            'timestamp': 1618123456789,
            'datetime': '2021-04-11T12:34:56.789Z'
        }
        
        self.sample_ohlcv = [
            [1618123200000, 57000, 60000, 55000, 58050, 1000],
            [1618126800000, 58050, 61000, 57500, 59000, 1200]
        ]
        
        # Mock cho CCXT
        self.async_exchange_mock = MagicMock()
        self.exchange_mock = MagicMock()
        
        # Patch cho ccxt.async_support và ccxt
        self.patch_ccxt_async = patch('ccxt.async_support.binance', return_value=self.async_exchange_mock)
        self.patch_ccxt = patch('ccxt.binance', return_value=self.exchange_mock)
        
        # Start patches
        self.mock_ccxt_async = self.patch_ccxt_async.start()
        self.mock_ccxt = self.patch_ccxt.start()
        
        # Mock các phương thức async
        self.async_exchange_mock.load_markets = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.load_markets.return_value.set_result({'BTC/USDT': {}})
        
        self.async_exchange_mock.fetch_time = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.fetch_time.return_value.set_result(1618123456789)
        
        # Khởi tạo connector với các mock
        self.connector = BinanceConnector(
            api_key="test_key",
            api_secret="test_secret",
            sandbox=True,
            config=self.config,
            secret_manager=self.secret_manager
        )
        
        # Mock các phương thức API của Binance
        self.async_exchange_mock.publicGetExchangeInfo = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.publicGetExchangeInfo.return_value.set_result({
            'symbols': [{'symbol': 'BTCUSDT', 'status': 'TRADING'}]
        })
        
        self.async_exchange_mock.privateGetTradeFee = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.privateGetTradeFee.return_value.set_result({
            'tradeFee': [{'symbol': 'BTCUSDT', 'maker': 0.001, 'taker': 0.001}]
        })
        
        self.async_exchange_mock.privateGetAccount = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.privateGetAccount.return_value.set_result({
            'accountType': 'SPOT'
        })
    
    def tearDown(self):
        """Clean up after each test."""
        # Stop patches
        self.patch_ccxt_async.stop()
        self.patch_ccxt.stop()
        
        # Clean up event loop
        tasks = asyncio.all_tasks(loop=asyncio.get_event_loop())
        for task in tasks:
            task.cancel()
    
    def test_initialization(self):
        """Test BinanceConnector initialization."""
        self.assertEqual(self.connector.exchange_id, 'binance')
        self.assertEqual(self.connector.api_key, 'test_key')
        self.assertEqual(self.connector.api_secret, 'test_secret')
        self.assertTrue(self.connector.exchange.sandbox)
        self.assertTrue(self.connector.async_exchange.sandbox)

    @patch('websockets.connect')
    @unittest.skip("Skip websocket test")
    def test_subscribe_to_websocket(self, mock_connect):
        """Test subscribing to Binance websocket."""
        # Cài đặt mock cho websocket
        mock_ws = MagicMock()
        mock_connect.return_value.__aenter__.return_value = mock_ws
        mock_ws.recv = MagicMock(return_value=asyncio.Future())
        mock_ws.recv.return_value.set_result(json.dumps({
            'stream': 'btcusdt@ticker',
            'data': {'s': 'BTCUSDT', 'c': '58050'}
        }))
        
        # Test subscribe_to_websocket
        async def test_subscribe():
            await self.connector.subscribe_to_websocket('BTC/USDT', ['ticker'])
            # Kiểm tra xem kết nối websocket đã được thiết lập chưa
            await asyncio.sleep(0.1)  # Chờ một chút để task websocket chạy
            mock_connect.assert_called_once()
        
        asyncio.run(test_subscribe())

    def test_get_exchange_info(self):
        """Test getting exchange info from Binance."""
        async def test_get_info():
            exchange_info = await self.connector.get_exchange_info()
            self.assertIn('symbols', exchange_info)
            self.assertEqual(len(exchange_info['symbols']), 1)
            self.assertEqual(exchange_info['symbols'][0]['symbol'], 'BTCUSDT')
        
        asyncio.run(test_get_info())

    def test_get_trading_fees(self):
        """Test getting trading fees from Binance."""
        async def test_fees():
            fees = await self.connector.get_trading_fees()
            self.assertIn('BTCUSDT', fees)
            self.assertEqual(fees['BTCUSDT']['maker'], 0.001)
            self.assertEqual(fees['BTCUSDT']['taker'], 0.001)
        
        asyncio.run(test_fees())

    def test_fetch_ticker(self):
        """Test fetching ticker from Binance."""
        # Setup mock
        self.async_exchange_mock.fetch_ticker = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.fetch_ticker.return_value.set_result(self.sample_ticker)
        
        async def test_ticker():
            ticker = await self.connector.fetch_ticker('BTC/USDT')
            self.assertEqual(ticker['symbol'], 'BTC/USDT')
            self.assertEqual(ticker['last'], 58050)
            self.async_exchange_mock.fetch_ticker.assert_called_once_with('BTC/USDT')
        
        asyncio.run(test_ticker())

    def test_fetch_order_book(self):
        """Test fetching order book from Binance."""
        # Setup mock
        self.async_exchange_mock.fetch_order_book = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.fetch_order_book.return_value.set_result(self.sample_orderbook)
        
        async def test_orderbook():
            orderbook = await self.connector.fetch_order_book('BTC/USDT', 20)
            self.assertEqual(orderbook['symbol'], 'BTC/USDT')
            self.assertEqual(len(orderbook['bids']), 2)
            self.assertEqual(len(orderbook['asks']), 2)
            self.async_exchange_mock.fetch_order_book.assert_called_once_with('BTC/USDT', 20)
        
        asyncio.run(test_orderbook())

    def test_fetch_ohlcv(self):
        """Test fetching OHLCV data from Binance."""
        # Setup mock
        self.async_exchange_mock.fetch_ohlcv = MagicMock(return_value=asyncio.Future())
        self.async_exchange_mock.fetch_ohlcv.return_value.set_result(self.sample_ohlcv)
        
        async def test_ohlcv():
            ohlcv = await self.connector.fetch_ohlcv('BTC/USDT', '1h', None, 2)
            self.assertEqual(len(ohlcv), 2)
            self.assertEqual(ohlcv[0][4], 58050)  # Closing price
            self.async_exchange_mock.fetch_ohlcv.assert_called_once_with('BTC/USDT', '1h', None, 2)
        
        asyncio.run(test_ohlcv())

    @patch('data_collectors.exchange_api.binance_connector.BinanceConnector.set_leverage')
    @unittest.skip("Skip futures test")
    def test_set_leverage(self, mock_set_leverage):
        """Test setting leverage for futures trading."""
        # Setup mock
        mock_set_leverage.return_value = asyncio.Future()
        mock_set_leverage.return_value.set_result({'symbol': 'BTCUSDT', 'leverage': 10})
        
        # Switch to futures mode
        self.connector.futures = True
        
        async def test_leverage():
            result = await self.connector.set_leverage('BTC/USDT', 10)
            self.assertEqual(result['symbol'], 'BTCUSDT')
            self.assertEqual(result['leverage'], 10)
            mock_set_leverage.assert_called_once_with('BTC/USDT', 10)
        
        asyncio.run(test_leverage())


if __name__ == '__main__':
    unittest.main()