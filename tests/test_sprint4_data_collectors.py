"""
Test các module thu thập dữ liệu thị trường từ Sprint 4.
File này chứa các test case để kiểm tra các lớp và chức năng của:
- historical_data_collector.py
- realtime_data_stream.py
- orderbook_collector.py
"""

import os
import sys
import pytest
import json
import pandas as pd
import numpy as np
import asyncio
import collections
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from unittest.mock import MagicMock, patch, AsyncMock, Mock

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import các module cần test
from data_collectors.market_data.historical_data_collector import (
    HistoricalDataCollector, 
    create_data_collector
)
from data_collectors.market_data.realtime_data_stream import (
    RealtimeDataStream, 
    DataHandler, 
    ConsoleOutputHandler,
    CSVStorageHandler, 
    CustomProcessingHandler,
    create_realtime_stream
)
from data_collectors.market_data.orderbook_collector import (
    OrderbookSnapshot, 
    OrderbookManager, 
    OrderbookCollector, 
    MarketLiquidityMonitor,
    create_orderbook_collector
)

# Import các dependencies
from data_collectors.exchange_api.generic_connector import ExchangeConnector
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.bybit_connector import BybitConnector

# =================== FIXTURES ===================

@pytest.fixture
def sample_orderbook_data():
    """Fixture cung cấp dữ liệu orderbook mẫu"""
    return {
        'bids': [[10000.0, 1.5], [9990.0, 2.0], [9980.0, 2.5]],
        'asks': [[10010.0, 1.0], [10020.0, 2.0], [10030.0, 3.0]],
        'timestamp': int(datetime.now().timestamp() * 1000),
        'nonce': 12345
    }

@pytest.fixture
def sample_ohlcv_data():
    """Fixture cung cấp dữ liệu OHLCV mẫu"""
    now = int(datetime.now().timestamp() * 1000)
    return [
        [now - 60000, 10000.0, 10050.0, 9950.0, 10025.0, 10.5],
        [now, 10025.0, 10075.0, 10000.0, 10050.0, 12.3]
    ]

@pytest.fixture
def sample_trades_data():
    """Fixture cung cấp dữ liệu giao dịch mẫu"""
    now = int(datetime.now().timestamp() * 1000)
    return [
        {
            'id': '123456',
            'timestamp': now - 30000,
            'datetime': datetime.fromtimestamp((now - 30000) / 1000).isoformat(),
            'symbol': 'BTC/USDT',
            'side': 'buy',
            'price': 10025.0,
            'amount': 0.5,
            'cost': 5012.5
        },
        {
            'id': '123457',
            'timestamp': now,
            'datetime': datetime.fromtimestamp(now / 1000).isoformat(),
            'symbol': 'BTC/USDT',
            'side': 'sell',
            'price': 10050.0,
            'amount': 0.7,
            'cost': 7035.0
        }
    ]

@pytest.fixture
def mock_exchange_connector():
    """Fixture cung cấp mock exchange connector cho tests"""
    connector = AsyncMock(spec=ExchangeConnector)
    connector.exchange_id = 'binance'
    connector.markets = {'BTC/USDT': {}, 'ETH/USDT': {}}
    
    # Mock phương thức fetch_order_book để trả về dữ liệu orderbook mẫu
    async def mock_fetch_order_book(symbol, limit=None):
        return {
            'bids': [[10000.0, 1.5], [9990.0, 2.0], [9980.0, 2.5]],
            'asks': [[10010.0, 1.0], [10020.0, 2.0], [10030.0, 3.0]],
            'timestamp': int(datetime.now().timestamp() * 1000),
            'nonce': 12345
        }
    connector.fetch_order_book = mock_fetch_order_book
    
    # Mock phương thức fetch_ohlcv để trả về dữ liệu OHLCV mẫu
    async def mock_fetch_ohlcv(symbol, timeframe, since=None, limit=None):
        now = int(datetime.now().timestamp() * 1000)
        return [
            [now - 60000, 10000.0, 10050.0, 9950.0, 10025.0, 10.5],
            [now, 10025.0, 10075.0, 10000.0, 10050.0, 12.3]
        ]
    connector.fetch_ohlcv = mock_fetch_ohlcv
    
    # Mock phương thức fetch_trades để trả về dữ liệu giao dịch mẫu
    async def mock_fetch_trades(symbol, since=None, limit=None):
        now = int(datetime.now().timestamp() * 1000)
        return [
            {
                'id': '123456',
                'timestamp': now - 30000,
                'datetime': datetime.fromtimestamp((now - 30000) / 1000).isoformat(),
                'symbol': symbol,
                'side': 'buy',
                'price': 10025.0,
                'amount': 0.5,
                'cost': 5012.5
            },
            {
                'id': '123457',
                'timestamp': now,
                'datetime': datetime.fromtimestamp(now / 1000).isoformat(),
                'symbol': symbol,
                'side': 'sell',
                'price': 10050.0,
                'amount': 0.7,
                'cost': 7035.0
            }
        ]
    connector.fetch_trades = mock_fetch_trades
    
    # Mock exchange để có rate_limit
    connector.exchange = MagicMock()
    connector.exchange.rateLimit = 1000  # 1 giây
    
    return connector

@pytest.fixture
def temp_data_dir(tmp_path):
    """Fixture cung cấp thư mục tạm để lưu dữ liệu"""
    data_dir = tmp_path / 'data'
    data_dir.mkdir()
    return data_dir

# =============== TESTS FOR ORDERBOOK COLLECTOR ===============

class TestOrderbookSnapshot:
    """Kiểm tra lớp OrderbookSnapshot"""
    
    def test_init_and_properties(self):
        """Kiểm tra khởi tạo và các thuộc tính"""
        snapshot = OrderbookSnapshot(
            symbol='BTC/USDT',
            timestamp=1615300800000,
            bids=[[10000.0, 1.5], [9990.0, 2.0]],
            asks=[[10010.0, 1.0], [10020.0, 2.0]],
            last_update_id=12345
        )
        
        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.timestamp == 1615300800000
        assert len(snapshot.bids) == 2
        assert len(snapshot.asks) == 2
        assert snapshot.last_update_id == 12345
        
        # Kiểm tra bids đã được sắp xếp giảm dần theo giá
        assert snapshot.bids[0][0] > snapshot.bids[1][0]
        
        # Kiểm tra asks đã được sắp xếp tăng dần theo giá
        assert snapshot.asks[0][0] < snapshot.asks[1][0]
    
    def test_validation(self):
        """Kiểm tra validation khi khởi tạo"""
        # Symbol không hợp lệ
        with pytest.raises(ValueError):
            OrderbookSnapshot(symbol='')
        
        # Bids không hợp lệ
        with pytest.raises(ValueError):
            OrderbookSnapshot(
                symbol='BTC/USDT',
                bids=[['not_a_number', 1.0]]
            )
        
        # Asks không hợp lệ
        with pytest.raises(ValueError):
            OrderbookSnapshot(
                symbol='BTC/USDT',
                asks=[[10000.0]]  # Thiếu khối lượng
            )
    
    def test_get_mid_price(self):
        """Kiểm tra tính giá trung bình"""
        snapshot = OrderbookSnapshot(
            symbol='BTC/USDT',
            bids=[[10000.0, 1.0]],
            asks=[[10100.0, 1.0]]
        )
        
        # Mid price = (10000 + 10100) / 2 = 10050
        assert snapshot.get_mid_price() == 10050.0
        
        # Trường hợp không có bids hoặc asks
        empty_snapshot = OrderbookSnapshot(symbol='BTC/USDT')
        assert empty_snapshot.get_mid_price() == 0.0
    
    def test_get_spread(self):
        """Kiểm tra tính spread"""
        snapshot = OrderbookSnapshot(
            symbol='BTC/USDT',
            bids=[[10000.0, 1.0]],
            asks=[[10100.0, 1.0]]
        )
        
        # Spread = 10100 - 10000 = 100
        assert snapshot.get_spread() == 100.0
        
        # Spread percentage = (100 / 10050) * 100 = 0.995%
        assert round(snapshot.get_spread_percentage(), 3) == 0.995
    
    def test_get_liquidity_within_range(self):
        """Kiểm tra tính thanh khoản trong khoảng"""
        snapshot = OrderbookSnapshot(
            symbol='BTC/USDT',
            bids=[[10000.0, 1.0], [9900.0, 2.0]],
            asks=[[10100.0, 1.5], [10200.0, 2.5]]
        )
        
        # Mid price = 10050, khoảng 1% = [9949.5, 10150.5]
        liquidity = snapshot.get_liquidity_within_range(1.0)
        
        # Chỉ có bid[0] trong khoảng
        assert liquidity['bid_volume'] == 1.0
        assert liquidity['bid_value'] == 10000.0
        
        # Chỉ có ask[0] trong khoảng
        assert liquidity['ask_volume'] == 1.5
        assert liquidity['ask_value'] == 15150.0
    
    def test_get_imbalance(self):
        """Kiểm tra tính mất cân bằng"""
        # Trường hợp thiên về bên mua
        snapshot1 = OrderbookSnapshot(
            symbol='BTC/USDT',
            bids=[[10000.0, 3.0]],
            asks=[[10100.0, 1.0]]
        )
        
        # imbalance = (3.0 - 1.0) / (3.0 + 1.0) = 0.5
        assert snapshot1.get_imbalance() == 0.5
        
        # Trường hợp thiên về bên bán
        snapshot2 = OrderbookSnapshot(
            symbol='BTC/USDT',
            bids=[[10000.0, 1.0]],
            asks=[[10100.0, 3.0]]
        )
        
        # imbalance = (1.0 - 3.0) / (1.0 + 3.0) = -0.5
        assert snapshot2.get_imbalance() == -0.5
    
    def test_to_dict_and_from_dict(self):
        """Kiểm tra chuyển đổi giữa snapshot và dict"""
        original = OrderbookSnapshot(
            symbol='BTC/USDT',
            timestamp=1615300800000,
            bids=[[10000.0, 1.5]],
            asks=[[10100.0, 1.0]],
            last_update_id=12345
        )
        
        # Chuyển từ snapshot sang dict
        data_dict = original.to_dict()
        
        # Chuyển từ dict trở lại snapshot
        recreated = OrderbookSnapshot.from_dict(data_dict)
        
        # Kiểm tra các thuộc tính quan trọng
        assert recreated.symbol == original.symbol
        assert recreated.timestamp == original.timestamp
        assert recreated.bids == original.bids
        assert recreated.asks == original.asks
        assert recreated.last_update_id == original.last_update_id
    
    def test_from_ccxt_format(self, sample_orderbook_data):
        """Kiểm tra tạo snapshot từ định dạng CCXT"""
        snapshot = OrderbookSnapshot.from_ccxt_format('BTC/USDT', sample_orderbook_data)
        
        assert snapshot.symbol == 'BTC/USDT'
        assert snapshot.timestamp == sample_orderbook_data['timestamp']
        assert snapshot.bids == sample_orderbook_data['bids']
        assert snapshot.asks == sample_orderbook_data['asks']
        assert snapshot.last_update_id == sample_orderbook_data['nonce']


class TestOrderbookManager:
    """Kiểm tra lớp OrderbookManager"""
    
    def test_init(self):
        """Kiểm tra khởi tạo"""
        manager = OrderbookManager('BTC/USDT', max_depth=50, buffer_size=5)
        
        assert manager.symbol == 'BTC/USDT'
        assert manager.max_depth == 50
        assert manager.buffer_size == 5
        assert isinstance(manager.current_orderbook, OrderbookSnapshot)
        assert len(manager.orderbook_buffer) == 0
        assert manager.is_synced == False
    
    def test_update_from_snapshot(self, sample_orderbook_data):
        """Kiểm tra cập nhật từ snapshot"""
        manager = OrderbookManager('BTC/USDT')
        
        # Tạo snapshot
        snapshot = OrderbookSnapshot.from_ccxt_format('BTC/USDT', sample_orderbook_data)
        
        # Cập nhật manager
        result = manager.update_from_snapshot(snapshot)
        
        assert result == True
        assert manager.is_synced == True
        assert manager.last_update_id == sample_orderbook_data['nonce']
        assert manager.current_orderbook.bids == snapshot.bids
        assert manager.current_orderbook.asks == snapshot.asks
        
        # Kiểm tra orderbook buffer
        assert len(manager.orderbook_buffer) == 1
    
    def test_update_from_delta(self, sample_orderbook_data):
        """Kiểm tra cập nhật từ delta"""
        manager = OrderbookManager('BTC/USDT')
        
        # Khởi tạo với snapshot đầu tiên
        snapshot = OrderbookSnapshot.from_ccxt_format('BTC/USDT', sample_orderbook_data)
        manager.update_from_snapshot(snapshot)
        
        # Tạo delta mới
        delta = {
            'lastUpdateId': sample_orderbook_data['nonce'] + 1,
            'bids': [[9995.0, 1.8]],  # Thêm mức giá mới
            'asks': [[10010.0, 0]]    # Xóa mức giá cũ
        }
        
        # Cập nhật từ delta
        result = manager.update_from_delta(delta)
        
        assert result == True
        
        # Kiểm tra bids đã được cập nhật
        updated_bids = manager.current_orderbook.bids
        assert any(bid[0] == 9995.0 and bid[1] == 1.8 for bid in updated_bids)
        
        # Kiểm tra asks đã được cập nhật (mức giá 10010.0 đã bị xóa)
        updated_asks = manager.current_orderbook.asks
        assert not any(ask[0] == 10010.0 for ask in updated_asks)
    
    def test_get_vwap(self, sample_orderbook_data):
        """Kiểm tra tính VWAP (Volume Weighted Average Price)"""
        manager = OrderbookManager('BTC/USDT')
        
        # Khởi tạo với snapshot
        snapshot = OrderbookSnapshot.from_ccxt_format('BTC/USDT', sample_orderbook_data)
        manager.update_from_snapshot(snapshot)
        
        # Tính VWAP để mua 3.0 BTC
        vwap = manager.get_vwap(3.0, 'asks')
        
        # VWAP = (10010*1.0 + 10020*2.0) / 3.0 = (10010 + 20040) / 3 = 10016.67
        assert abs(vwap - 10016.67) < 0.01
        
        # Tính VWAP để bán 3.0 BTC
        vwap = manager.get_vwap(3.0, 'bids')
        
        # VWAP = (10000*1.5 + 9990*1.5) / 3.0 = (15000 + 14985) / 3 = 9995
        assert abs(vwap - 9995.0) < 0.01
    
    def test_get_executed_price(self, sample_orderbook_data):
        """Kiểm tra tính giá thực thi"""
        manager = OrderbookManager('BTC/USDT')
        
        # Khởi tạo với snapshot
        snapshot = OrderbookSnapshot.from_ccxt_format('BTC/USDT', sample_orderbook_data)
        manager.update_from_snapshot(snapshot)
        
        # Giá thực thi khi mua 2.0 BTC
        buy_price = manager.get_executed_price(2.0, 'buy')
        
        # = (10010*1.0 + 10020*1.0) / 2.0 = 10015
        assert abs(buy_price - 10015.0) < 0.01
        
        # Giá thực thi khi bán 2.0 BTC
        sell_price = manager.get_executed_price(2.0, 'sell')
        
        # = (10000*1.5 + 9990*0.5) / 2.0 = 9997.5
        assert abs(sell_price - 9997.5) < 0.01


@pytest.mark.asyncio
class TestOrderbookCollector:
    """Kiểm tra lớp OrderbookCollector"""
    
    async def test_init(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra khởi tạo"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir,
            snapshot_interval=30,
            max_depth=50,
            buffer_size=5,
            max_file_snapshots=100
        )
        
        assert collector.exchange_id == 'binance'
        assert collector.snapshot_interval == 30
        assert collector.max_depth == 50
        assert collector.buffer_size == 5
        assert collector.max_file_snapshots == 100
        assert len(collector.orderbook_managers) == 0
        assert collector.is_running == False
    
    async def test_initialize_orderbook(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra khởi tạo orderbook cho một cặp giao dịch"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        # Khởi tạo orderbook cho BTC/USDT
        success = await collector.initialize_orderbook('BTC/USDT')
        
        assert success == True
        assert 'BTC/USDT' in collector.orderbook_managers
        assert collector.orderbook_managers['BTC/USDT'].is_synced == True
        
        # Thử khởi tạo với symbol không hợp lệ
        success = await collector.initialize_orderbook('INVALID/PAIR')
        assert success == False
    
    async def test_update_from_websocket(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra cập nhật từ websocket"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        # Khởi tạo orderbook
        await collector.initialize_orderbook('BTC/USDT')
        
        # Tạo dữ liệu websocket snapshot
        websocket_snapshot = {
            'type': 'orderbook',
            'symbol': 'BTC/USDT',
            'timestamp': int(datetime.now().timestamp() * 1000),
            'bids': [[10005.0, 1.2], [9995.0, 2.2]],
            'asks': [[10015.0, 0.8], [10025.0, 1.8]],
            'nonce': 12346
        }
        
        # Cập nhật từ websocket
        result = await collector.update_from_websocket(websocket_snapshot)
        
        assert result == True
        
        # Kiểm tra orderbook đã được cập nhật
        snapshot = collector.get_orderbook('BTC/USDT')
        assert snapshot.last_update_id == 12346
        assert any(bid[0] == 10005.0 for bid in snapshot.bids)
        assert any(ask[0] == 10015.0 for ask in snapshot.asks)
    
    async def test_start_and_stop(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra bắt đầu và dừng thu thập dữ liệu"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir,
            snapshot_interval=1  # Để test nhanh hơn
        )
        
        # Bắt đầu thu thập
        await collector.start(['BTC/USDT', 'ETH/USDT'])
        
        assert collector.is_running == True
        assert len(collector.tasks) == 2
        assert 'BTC/USDT' in collector.orderbook_managers
        assert 'ETH/USDT' in collector.orderbook_managers
        
        # Đợi một chút để thu thập dữ liệu
        await asyncio.sleep(1.2)
        
        # Dừng thu thập
        await collector.stop()
        
        assert collector.is_running == False
        assert len(collector.tasks) == 0
    
    async def test_callbacks(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra cơ chế callback"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        # Tạo mock callbacks
        update_callback = AsyncMock()
        snapshot_callback = AsyncMock()
        
        # Đăng ký callbacks
        collector.add_update_callback(update_callback)
        collector.add_snapshot_callback(snapshot_callback)
        
        # Khởi tạo orderbook
        await collector.initialize_orderbook('BTC/USDT')
        
        # Kiểm tra callbacks đã được gọi
        assert update_callback.call_count >= 1
        assert snapshot_callback.call_count >= 1
        
        # Kiểm tra tham số của callbacks
        update_args = update_callback.call_args[0]
        assert update_args[0] == 'BTC/USDT'  # symbol
        assert isinstance(update_args[1], dict)  # metrics
        
        snapshot_args = snapshot_callback.call_args[0]
        assert snapshot_args[0] == 'BTC/USDT'  # symbol
        assert isinstance(snapshot_args[1], OrderbookSnapshot)  # snapshot


@pytest.mark.asyncio
class TestMarketLiquidityMonitor:
    """Kiểm tra lớp MarketLiquidityMonitor"""
    
    async def test_init(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra khởi tạo"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        monitor = MarketLiquidityMonitor(
            orderbook_collector=collector,
            alert_threshold=10.0,
            window_size=5
        )
        
        assert monitor.orderbook_collector == collector
        assert monitor.alert_threshold == 10.0
        assert monitor.window_size == 5
        assert len(monitor.liquidity_history) == 0
        assert len(monitor._alert_callbacks) == 0
    
    async def test_on_orderbook_update(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra xử lý update orderbook"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        monitor = MarketLiquidityMonitor(
            orderbook_collector=collector,
            alert_threshold=10.0,
            window_size=5
        )
        
        # Gửi update giả lập
        metrics = {
            'symbol': 'BTC/USDT',
            'timestamp': int(datetime.now().timestamp() * 1000),
            'mid_price': 10050.0,
            'spread': 100.0,
            'spread_percentage': 1.0,
            'bid_ask_ratio': 1.2,
            'imbalance': 0.1,
            'bid_volume_1pct': 5.0,
            'ask_volume_1pct': 4.0,
            'total_volume_1pct': 9.0
        }
        
        await monitor._on_orderbook_update('BTC/USDT', metrics)
        
        # Kiểm tra metrics đã được lưu vào history
        assert 'BTC/USDT' in monitor.liquidity_history
        assert len(monitor.liquidity_history['BTC/USDT']) == 1
        assert monitor.liquidity_history['BTC/USDT'][0] == metrics
    
    async def test_analyze_trends_with_alert(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra phát hiện biến động và gửi cảnh báo"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        monitor = MarketLiquidityMonitor(
            orderbook_collector=collector,
            alert_threshold=5.0,  # Ngưỡng thấp để dễ kích hoạt cảnh báo
            window_size=5
        )
        
        # Đăng ký mock callback
        alert_callback = AsyncMock()
        monitor.add_alert_callback(alert_callback)
        
        # Tạo metrics trước đó
        previous_metrics = {
            'symbol': 'BTC/USDT',
            'timestamp': int((datetime.now() - timedelta(minutes=1)).timestamp() * 1000),
            'mid_price': 10000.0,
            'spread': 100.0,
            'spread_percentage': 1.0,
            'imbalance': 0.1,
            'bid_volume_1pct': 5.0,
            'ask_volume_1pct': 5.0,
            'total_volume_1pct': 10.0
        }
        
        # Tạo metrics hiện tại với biến động lớn
        latest_metrics = {
            'symbol': 'BTC/USDT',
            'timestamp': int(datetime.now().timestamp() * 1000),
            'mid_price': 10600.0,  # Tăng 6%
            'spread': 100.0,
            'spread_percentage': 1.0,
            'imbalance': 0.1,
            'bid_volume_1pct': 5.0,
            'ask_volume_1pct': 5.0,
            'total_volume_1pct': 10.0
        }
        
        # Thêm vào history
        monitor.liquidity_history['BTC/USDT'] = collections.deque(maxlen=5)
        monitor.liquidity_history['BTC/USDT'].append(previous_metrics)
        monitor.liquidity_history['BTC/USDT'].append(latest_metrics)
        
        # Phân tích xu hướng
        await monitor._analyze_trends('BTC/USDT')
        
        # Kiểm tra callback đã được gọi
        assert alert_callback.call_count == 1
        
        # Kiểm tra dữ liệu cảnh báo
        alert_data = alert_callback.call_args[0][1]
        assert alert_data['symbol'] == 'BTC/USDT'
        assert len(alert_data['alerts']) >= 1
        assert any(alert['type'] == 'price' for alert in alert_data['alerts'])
    
    def test_get_liquidity_trend(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra lấy xu hướng thanh khoản"""
        collector = OrderbookCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        monitor = MarketLiquidityMonitor(
            orderbook_collector=collector,
            alert_threshold=10.0,
            window_size=5
        )
        
        # Tạo dữ liệu cho history
        start_time = int((datetime.now() - timedelta(minutes=5)).timestamp() * 1000)
        end_time = int(datetime.now().timestamp() * 1000)
        
        monitor.liquidity_history['BTC/USDT'] = collections.deque(maxlen=5)
        monitor.liquidity_history['BTC/USDT'].append({
            'symbol': 'BTC/USDT',
            'timestamp': start_time,
            'mid_price': 10000.0,
            'spread_percentage': 1.0,
            'imbalance': 0.1,
            'bid_volume_1pct': 5.0,
            'ask_volume_1pct': 5.0
        })
        monitor.liquidity_history['BTC/USDT'].append({
            'symbol': 'BTC/USDT',
            'timestamp': end_time,
            'mid_price': 10500.0,
            'spread_percentage': 1.2,
            'imbalance': 0.2,
            'bid_volume_1pct': 6.0,
            'ask_volume_1pct': 4.5
        })
        
        # Lấy xu hướng
        trend = monitor.get_liquidity_trend('BTC/USDT')
        
        assert trend['has_data'] == True
        assert trend['symbol'] == 'BTC/USDT'
        assert trend['price']['first'] == 10000.0
        assert trend['price']['last'] == 10500.0
        assert abs(trend['price']['change_pct'] - 5.0) < 0.01
        assert trend['liquidity']['imbalance_change'] == 0.1

# =============== TESTS FOR HISTORICAL DATA COLLECTOR ===============

@pytest.mark.asyncio
class TestHistoricalDataCollector:
    """Kiểm tra lớp HistoricalDataCollector"""
    
    async def test_init(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra khởi tạo"""
        collector = HistoricalDataCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir,
            max_workers=4,
            rate_limit_factor=0.8
        )
        
        assert collector.exchange_connector == mock_exchange_connector
        assert collector.exchange_id == 'binance'
        assert collector.max_workers == 4
        assert collector.rate_limit_sleep > 0
        
        # Kiểm tra các thư mục đã được tạo
        assert (temp_data_dir / 'historical' / 'binance' / 'ohlcv').exists()
        assert (temp_data_dir / 'historical' / 'binance' / 'orderbook').exists()
        assert (temp_data_dir / 'historical' / 'binance' / 'trades').exists()
        assert (temp_data_dir / 'historical' / 'binance' / 'funding').exists()
    
    async def test_collect_ohlcv(self, mock_exchange_connector, temp_data_dir, sample_ohlcv_data):
        """Kiểm tra thu thập dữ liệu OHLCV"""
        collector = HistoricalDataCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        # Thu thập dữ liệu
        df = await collector.collect_ohlcv(
            symbol='BTC/USDT',
            timeframe='1h',
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            save_format='csv'  # Dùng CSV để dễ kiểm tra
        )
        
        # Kiểm tra DataFrame
        assert len(df) == 2
        assert list(df.columns) == ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        
        # Kiểm tra file đã được tạo
        assert (temp_data_dir / 'historical' / 'binance' / 'ohlcv' / 'btc_usdt_1h.csv').exists()
    
    async def test_collect_trades(self, mock_exchange_connector, temp_data_dir, sample_trades_data):
        """Kiểm tra thu thập dữ liệu giao dịch"""
        collector = HistoricalDataCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        # Thu thập dữ liệu
        df = await collector.collect_historical_trades(
            symbol='BTC/USDT',
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            save_format='csv'
        )
        
        # Kiểm tra DataFrame
        assert len(df) == 2
        assert 'id' in df.columns
        assert 'price' in df.columns
        assert 'amount' in df.columns
        assert 'side' in df.columns
        
        # Kiểm tra file đã được tạo
        assert (temp_data_dir / 'historical' / 'binance' / 'trades' / 'btc_usdt_trades.csv').exists()
    
    async def test_collect_all_symbols_ohlcv(self, mock_exchange_connector, temp_data_dir):
        """Kiểm tra thu thập dữ liệu OHLCV cho nhiều cặp giao dịch"""
        collector = HistoricalDataCollector(
            exchange_connector=mock_exchange_connector,
            data_dir=temp_data_dir
        )
        
        # Thu thập dữ liệu cho nhiều cặp giao dịch
        results = await collector.collect_all_symbols_ohlcv(
            symbols=['BTC/USDT', 'ETH/USDT'],
            timeframe='1h',
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now(),
            concurrency=2
        )
        
        # Kiểm tra kết quả
        assert 'BTC/USDT' in results
        assert 'ETH/USDT' in results
        assert len(results['BTC/USDT']) > 0
        assert len(results['ETH/USDT']) > 0
    
    def test_get_available_timeframes(self):
        """Kiểm tra lấy danh sách timeframes"""
        timeframes = HistoricalDataCollector.get_available_timeframes()
        
        assert '1m' in timeframes
        assert '1h' in timeframes
        assert '1d' in timeframes
        
        # Kiểm tra giá trị timeframes
        assert timeframes['1m'] == 60  # 1 phút = 60 giây
        assert timeframes['1h'] == 3600  # 1 giờ = 3600 giây

# =============== TESTS FOR REALTIME DATA STREAM ===============

class TestDataHandler:
    """Kiểm tra các lớp DataHandler"""
    
    def test_console_output_handler(self, caplog):
        """Kiểm tra ConsoleOutputHandler"""
        handler = ConsoleOutputHandler(name="test_console", log_level="INFO")
        
        # Tạo message test
        ticker_message = {
            'type': 'ticker',
            'symbol': 'BTC/USDT',
            'last': 10500.0,
            'bid': 10495.0,
            'ask': 10505.0
        }
        
        # Gọi process_data - sử dụng event loop để gọi hàm async
        asyncio.run(handler.process_data(ticker_message))
        
        # Kiểm tra log output
        assert "Ticker BTC/USDT: Last: 10500.0, Bid: 10495.0, Ask: 10505.0" in caplog.text
    
    def test_custom_processing_handler(self):
        """Kiểm tra CustomProcessingHandler"""
        # Tạo mock callback
        callback_mock = MagicMock()
        
        # Tạo filter function
        def filter_func(data):
            return data.get('type') == 'ticker'
        
        # Tạo handler
        handler = CustomProcessingHandler(
            callback=callback_mock,
            name="test_custom",
            filter_func=filter_func
        )
        
        # Tạo messages test
        ticker_message = {'type': 'ticker', 'symbol': 'BTC/USDT'}
        trade_message = {'type': 'trade', 'symbol': 'BTC/USDT'}
        
        # Gọi process_data cho cả hai messages
        asyncio.run(handler.process_data(ticker_message))
        asyncio.run(handler.process_data(trade_message))
        
        # Kiểm tra callback chỉ được gọi cho ticker message
        assert callback_mock.call_count == 1
        assert callback_mock.call_args[0][0] == ticker_message


@pytest.mark.asyncio
class TestRealtimeDataStream:
    """Kiểm tra lớp RealtimeDataStream"""
    
    async def test_init(self, mock_exchange_connector):
        """Kiểm tra khởi tạo"""
        # Tạo mock handlers
        handler1 = MagicMock(spec=DataHandler)
        handler1.name = "handler1"
        
        handler2 = MagicMock(spec=DataHandler)
        handler2.name = "handler2"
        
        # Tạo stream
        stream = RealtimeDataStream(
            exchange_connector=mock_exchange_connector,
            data_handlers=[handler1, handler2],
            websocket_reconnect_interval=15,
            heartbeat_interval=5
        )
        
        assert stream.exchange_connector == mock_exchange_connector
        assert stream.exchange_id == 'binance'
        assert len(stream.data_handlers) == 2
        assert stream.websocket_reconnect_interval == 15
        assert stream.heartbeat_interval == 5
        assert stream.is_running == False
    
    async def test_normalize_ws_message(self, mock_exchange_connector):
        """Kiểm tra chuẩn hóa thông điệp websocket"""
        stream = RealtimeDataStream(
            exchange_connector=mock_exchange_connector
        )
        
        # Tạo message Binance
        binance_ticker = {
            'stream': 'btcusdt@ticker',
            'data': {
                's': 'BTCUSDT',
                'E': 1615300800000,
                'c': '10500.0',
                'b': '10495.0',
                'a': '10505.0',
                'v': '1000.0',
                'h': '10600.0',
                'l': '10400.0',
                'p': '100.0',
                'P': '0.96',
                'o': '10400.0',
                'q': '10500000.0'
            }
        }
        
        # Chuẩn hóa message
        normalized = stream._normalize_binance_message(binance_ticker)
        
        # Kiểm tra kết quả
        assert normalized['type'] == 'ticker'
        assert normalized['symbol'] == 'BTC/USDT'
        assert normalized['timestamp'] == 1615300800000
        assert normalized['last'] == 10500.0
        assert normalized['bid'] == 10495.0
        assert normalized['ask'] == 10505.0
    
    async def test_process_message(self, mock_exchange_connector):
        """Kiểm tra xử lý thông điệp"""
        # Tạo mock handler
        handler = AsyncMock(spec=DataHandler)
        handler.name = "test_handler"
        handler.process_data = AsyncMock()
        
        # Tạo stream
        stream = RealtimeDataStream(
            exchange_connector=mock_exchange_connector,
            data_handlers=[handler]
        )
        
        # Khởi động stream
        await stream.start()
        
        # Tạo message
        message = {
            'type': 'ticker',
            'symbol': 'BTC/USDT',
            'timestamp': int(datetime.now().timestamp() * 1000),
            'last': 10500.0
        }
        
        # Xử lý message
        await stream._process_message(message)
        
        # Đợi queue được xử lý
        await asyncio.sleep(0.1)
        
        # Kiểm tra handler đã được gọi
        handler.process_data.assert_called_once()
        assert handler.process_data.call_args[0][0] == message
        
        # Dừng stream
        await stream.stop()
    
    async def test_subscribe_and_unsubscribe(self, mock_exchange_connector):
        """Kiểm tra đăng ký và hủy đăng ký"""
        # Mock hàm subscribe_to_websocket
        mock_exchange_connector.subscribe_to_websocket = AsyncMock()
        
        # Tạo stream
        stream = RealtimeDataStream(
            exchange_connector=mock_exchange_connector
        )
        
        # Bắt đầu stream
        await stream.start()
        
        # Đăng ký
        await stream.subscribe('BTC/USDT', ['ticker', 'trade'])
        
        # Kiểm tra đăng ký
        assert 'BTC/USDT' in stream.subscription_symbols
        assert stream.subscription_channels['BTC/USDT'] == ['ticker', 'trade']
        assert mock_exchange_connector.subscribe_to_websocket.called
        
        # Hủy đăng ký một kênh
        await stream.unsubscribe('BTC/USDT', ['trade'])
        
        # Kiểm tra hủy đăng ký
        assert 'BTC/USDT' in stream.subscription_symbols
        assert 'ticker' in stream.subscription_channels['BTC/USDT']
        assert 'trade' not in stream.subscription_channels['BTC/USDT']
        
        # Hủy đăng ký tất cả
        await stream.unsubscribe('BTC/USDT')
        
        # Kiểm tra hủy đăng ký hoàn toàn
        assert 'BTC/USDT' not in stream.subscription_symbols
        assert 'BTC/USDT' not in stream.subscription_channels
        
        # Dừng stream
        await stream.stop()

# =============== TESTS FOR FACTORY FUNCTIONS ===============

@pytest.mark.asyncio
class TestFactoryFunctions:
    """Kiểm tra các factory functions"""
    
    async def test_create_orderbook_collector(self, monkeypatch):
        """Kiểm tra create_orderbook_collector"""
        # Mock BinanceConnector
        mock_binance = AsyncMock(spec=BinanceConnector)
        mock_binance.exchange_id = 'binance'
        mock_binance.initialize = AsyncMock()
        
        # Patch constructor
        monkeypatch.setattr('data_collectors.market_data.orderbook_collector.BinanceConnector', MagicMock(return_value=mock_binance))
        
        # Tạo collector qua factory function
        collector = await create_orderbook_collector(
            exchange_id='binance',
            api_key='test_key',
            api_secret='test_secret',
            sandbox=True,
            snapshot_interval=30
        )
        
        # Kiểm tra các thuộc tính
        assert collector.exchange_connector == mock_binance
        assert collector.exchange_id == 'binance'
        assert collector.snapshot_interval == 30
        assert mock_binance.initialize.called
    
    async def test_create_data_collector(self, monkeypatch):
        """Kiểm tra create_data_collector"""
        # Mock BinanceConnector
        mock_binance = AsyncMock(spec=BinanceConnector)
        mock_binance.exchange_id = 'binance'
        mock_binance.initialize = AsyncMock()
        
        # Patch constructor
        monkeypatch.setattr('data_collectors.market_data.historical_data_collector.BinanceConnector', MagicMock(return_value=mock_binance))
        
        # Tạo collector qua factory function
        collector = await create_data_collector(
            exchange_id='binance',
            api_key='test_key',
            api_secret='test_secret',
            sandbox=True,
            max_workers=4
        )
        
        # Kiểm tra các thuộc tính
        assert collector.exchange_connector == mock_binance
        assert collector.exchange_id == 'binance'
        assert collector.max_workers == 4
        assert mock_binance.initialize.called
    
    async def test_create_realtime_stream(self, monkeypatch):
        """Kiểm tra create_realtime_stream"""
        # Mock BinanceConnector
        mock_binance = AsyncMock(spec=BinanceConnector)
        mock_binance.exchange_id = 'binance'
        mock_binance.initialize = AsyncMock()
        
        # Patch constructor
        monkeypatch.setattr('data_collectors.market_data.realtime_data_stream.BinanceConnector', MagicMock(return_value=mock_binance))
        
        # Tạo stream qua factory function
        stream = await create_realtime_stream(
            exchange_id='binance',
            api_key='test_key',
            api_secret='test_secret',
            sandbox=True
        )
        
        # Kiểm tra các thuộc tính
        assert stream.exchange_connector == mock_binance
        assert stream.exchange_id == 'binance'
        assert len(stream.data_handlers) >= 2  # Có ít nhất 2 handlers mặc định
        assert mock_binance.initialize.called
        
        # Kiểm tra các handlers mặc định
        assert any(isinstance(handler, ConsoleOutputHandler) for handler in stream.data_handlers)
        assert any(isinstance(handler, CSVStorageHandler) for handler in stream.data_handlers)


# Chạy tests khi được gọi trực tiếp
if __name__ == "__main__":
    pytest.main(["-xvs", __file__])