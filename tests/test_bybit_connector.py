"""
Unit tests for BybitConnector
"""

import os
import sys
import pytest
import logging
import asyncio
import json
from unittest.mock import patch, MagicMock, AsyncMock, call
import ccxt

# Đảm bảo đường dẫn đúng để import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup patching trước khi import
# Patch setup_logger trước khi import BybitConnector
patch('config.logging_config.setup_logger', return_value=logging.getLogger('test_mock')).start()
patch('config.security_config.SecretManager').start()
patch('config.system_config.SystemConfig').start()

# Import các module cần test sau khi đã patch
from data_collectors.exchange_api.bybit_connector import BybitConnector

# Logger cho tests
logger = logging.getLogger('test_bybit_connector')


@pytest.mark.asyncio
async def test_bybit_connector_init(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test khởi tạo BybitConnector."""
    logger.info("Testing BybitConnector initialization")
    
    # Test với tham số mặc định (linear futures)
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    assert connector.exchange_id == 'bybit'
    assert connector.api_key == 'test_api_key'
    assert connector.api_secret == 'test_api_secret'
    assert connector.category == 'linear'
    assert connector.sandbox is True
    assert connector.ws_manager is not None
    assert connector.exchange.options['defaultType'] == 'future'
    
    # Test với tham số category='spot'
    spot_connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        category='spot',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    assert spot_connector.category == 'spot'
    assert spot_connector.exchange.options['defaultType'] == 'spot'
    
    # Test với tham số category không hợp lệ (sẽ sử dụng 'linear' mặc định)
    invalid_connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        category='invalid',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    assert invalid_connector.category == 'linear'
    
    # Cleanup
    await connector.close()
    await spot_connector.close()
    await invalid_connector.close()
    
    logger.info("✅ BybitConnector initialization test passed")


@pytest.mark.asyncio
async def test_initialize(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test initialize method."""
    logger.info("Testing BybitConnector.initialize")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock các methods
    connector.get_exchange_info = AsyncMock(return_value=mock_exchange_data['bybit']['exchange_info'])
    connector.get_trading_fees = AsyncMock(return_value={'BTC/USDT': {'maker': 0.0002, 'taker': 0.0005}})
    connector._start_background_tasks = MagicMock()
    
    # Mock API call cho account verification
    account_info_response = {
        'retCode': 0,
        'retMsg': 'OK',
        'result': {'acctType': 'UNIFIED'}
    }
    connector.async_exchange.privateGetV5AccountWalletBalances = AsyncMock(return_value=account_info_response)
    
    # Test initialize với API key
    await connector.initialize()
    
    assert connector.markets is not None
    assert connector.get_exchange_info.called
    assert connector.get_trading_fees.called
    assert connector.async_exchange.privateGetV5AccountWalletBalances.called
    
    # Test initialize không có API key
    connector.api_key = None
    connector.api_secret = None
    connector.get_trading_fees.reset_mock()
    connector.async_exchange.privateGetV5AccountWalletBalances.reset_mock()
    
    await connector.initialize()
    
    assert not connector.get_trading_fees.called
    assert not connector.async_exchange.privateGetV5AccountWalletBalances.called
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.initialize test passed")


@pytest.mark.asyncio
async def test_get_exchange_info(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test get_exchange_info method."""
    logger.info("Testing BybitConnector.get_exchange_info")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock API call
    get_instruments_info = AsyncMock(return_value=mock_exchange_data['bybit']['exchange_info'])
    connector.async_exchange.publicGetV5MarketInstrumentsInfo = get_instruments_info
    
    # Test get_exchange_info
    exchange_info = await connector.get_exchange_info()
    
    assert exchange_info is not None
    assert get_instruments_info.called
    assert get_instruments_info.call_args[0][0] == {'category': 'linear'}
    
    # Test caching
    get_instruments_info.reset_mock()
    
    cached_exchange_info = await connector.get_exchange_info()
    
    assert cached_exchange_info is not None
    assert not get_instruments_info.called  # Should use cached value
    
    # Test error response
    get_instruments_info.reset_mock()
    get_instruments_info.return_value = {'retCode': 10001, 'retMsg': 'Error message'}
    
    # Invalidate cache
    connector.last_exchange_info_update = 0
    
    with pytest.raises(Exception):
        await connector.get_exchange_info()
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.get_exchange_info test passed")


@pytest.mark.asyncio
async def test_get_trading_fees(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test get_trading_fees method."""
    logger.info("Testing BybitConnector.get_trading_fees")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock API call
    fee_rate_response = {
        'retCode': 0,
        'retMsg': 'OK',
        'result': {
            'list': [
                {
                    'symbol': 'BTCUSDT',
                    'makerFeeRate': '0.0001',
                    'takerFeeRate': '0.0006'
                }
            ]
        }
    }
    get_fee_rate = AsyncMock(return_value=fee_rate_response)
    connector.async_exchange.privateGetV5AccountFeeRate = get_fee_rate
    
    # Test for specific symbol
    fees = await connector.get_trading_fees('BTC/USDT')
    
    assert fees is not None
    assert 'BTC/USDT' in fees
    assert fees['BTC/USDT']['maker'] == 0.0001
    assert fees['BTC/USDT']['taker'] == 0.0006
    assert get_fee_rate.called
    assert get_fee_rate.call_args[0][0] == {'category': 'linear', 'symbol': 'BTCUSDT'}
    
    # Test for all symbols
    get_fee_rate.reset_mock()
    fee_rate_all_response = {
        'retCode': 0,
        'retMsg': 'OK',
        'result': {
            'list': [
                {
                    'symbol': 'BTCUSDT',
                    'makerFeeRate': '0.0001',
                    'takerFeeRate': '0.0006'
                },
                {
                    'symbol': 'ETHUSDT',
                    'makerFeeRate': '0.0001',
                    'takerFeeRate': '0.0006'
                }
            ]
        }
    }
    get_fee_rate.return_value = fee_rate_all_response
    
    fees_all = await connector.get_trading_fees()
    
    assert fees_all is not None
    assert len(fees_all) == 2
    assert 'BTCUSDT' in fees_all
    assert 'ETHUSDT' in fees_all
    assert get_fee_rate.called
    assert get_fee_rate.call_args[0][0] == {'category': 'linear'}
    
    # Test with no API key
    connector.api_key = None
    
    with pytest.raises(ValueError):
        await connector.get_trading_fees()
    
    # Test error response
    connector.api_key = 'test_api_key'
    get_fee_rate.return_value = {'retCode': 10001, 'retMsg': 'Error message'}
    
    with pytest.raises(Exception):
        await connector.get_trading_fees()
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.get_trading_fees test passed")


@pytest.mark.asyncio
async def test_subscribe_to_websocket(patch_websockets, mock_system_config, mock_secret_manager):
    """Test subscribe_to_websocket method."""
    logger.info("Testing BybitConnector.subscribe_to_websocket")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock WebSocketManager and _generate_topics
    connector.ws_manager.add_connection = AsyncMock()
    connector._generate_topics = MagicMock(return_value=['tickers.BTCUSDT', 'kline.1.BTCUSDT'])
    
    # Test subscription
    await connector.subscribe_to_websocket('BTC/USDT', ['ticker', 'kline.1'])
    
    assert connector.ws_manager.add_connection.called
    call_args = connector.ws_manager.add_connection.call_args[0]
    assert call_args[0] == connector.V5_PUBLIC_WS_TESTNET_URL
    assert call_args[1] == ['BTCUSDT']
    assert call_args[2] == ['ticker', 'kline.1']
    
    # Test with empty topics
    connector._generate_topics.return_value = []
    connector.ws_manager.add_connection.reset_mock()
    
    await connector.subscribe_to_websocket('BTC/USDT', ['invalid_channel'])
    
    assert not connector.ws_manager.add_connection.called
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.subscribe_to_websocket test passed")


@pytest.mark.asyncio
async def test_generate_topics():
    """Test _generate_topics method."""
    logger.info("Testing BybitConnector._generate_topics")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret'
    )
    
    # Test with various channels
    symbols = ['BTCUSDT', 'ETHUSDT']
    channels = ['ticker', 'kline.1', 'orderbook', 'trade']
    
    topics = connector._generate_topics(symbols, channels)
    
    assert len(topics) == 8  # 2 symbols * 4 channels
    assert 'tickers.BTCUSDT' in topics
    assert 'tickers.ETHUSDT' in topics
    assert 'kline.1.BTCUSDT' in topics
    assert 'kline.1.ETHUSDT' in topics
    assert 'orderbook.50.BTCUSDT' in topics
    assert 'orderbook.50.ETHUSDT' in topics
    assert 'publicTrade.BTCUSDT' in topics
    assert 'publicTrade.ETHUSDT' in topics
    
    # Test with unsupported channel
    channels = ['unsupported']
    
    topics = connector._generate_topics(symbols, channels)
    
    assert len(topics) == 0
    
    # Test with mixed channels
    channels = ['ticker', 'unsupported']
    
    topics = connector._generate_topics(symbols, channels)
    
    assert len(topics) == 2
    assert 'tickers.BTCUSDT' in topics
    assert 'tickers.ETHUSDT' in topics
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector._generate_topics test passed")


@pytest.mark.asyncio
async def test_subscribe_multiple_symbols(patch_websockets, mock_system_config, mock_secret_manager):
    """Test subscribe_multiple_symbols method."""
    logger.info("Testing BybitConnector.subscribe_multiple_symbols")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock WebSocketManager
    connector.ws_manager.add_connection = AsyncMock()
    connector.ws_manager.symbols_per_connection = 2  # Force small batches for testing
    
    # Mock _generate_topics to return expected topics
    def mock_generate_topics(symbols, channels):
        result = []
        for symbol in symbols:
            for channel in channels:
                if channel == 'ticker':
                    result.append(f"tickers.{symbol}")
        return result
    
    connector._generate_topics = MagicMock(side_effect=mock_generate_topics)
    
    # Test multiple symbols subscription
    symbols = ['BTC/USDT', 'ETH/USDT', 'ADA/USDT', 'SOL/USDT']
    channels = ['ticker']
    
    await connector.subscribe_multiple_symbols(symbols, channels)
    
    # Should have called add_connection twice (2 symbols per batch)
    assert connector.ws_manager.add_connection.call_count == 2
    
    # Check first batch
    first_call_args = connector.ws_manager.add_connection.call_args_list[0][0]
    assert set(first_call_args[1]) == set(['BTCUSDT', 'ETHUSDT'])
    
    # Check second batch
    second_call_args = connector.ws_manager.add_connection.call_args_list[1][0]
    assert set(second_call_args[1]) == set(['ADAUSDT', 'SOLUSDT'])
    
    # Test with empty symbols list
    connector.ws_manager.add_connection.reset_mock()
    
    await connector.subscribe_multiple_symbols([], channels)
    
    assert not connector.ws_manager.add_connection.called
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.subscribe_multiple_symbols test passed")


@pytest.mark.asyncio
async def test_connect_bybit_websocket(patch_websockets, mock_system_config, mock_secret_manager):
    """Test _connect_bybit_websocket method."""
    logger.info("Testing BybitConnector._connect_bybit_websocket")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock _generate_topics
    connector._generate_topics = MagicMock(return_value=['tickers.BTCUSDT', 'kline.1.BTCUSDT'])
    
    # Test new connection
    websocket = await connector._connect_bybit_websocket(
        None, ['BTCUSDT'], ['ticker', 'kline.1']
    )
    
    assert websocket is not None
    assert websocket.send.called
    send_data = json.loads(websocket.send.call_args[0][0])
    assert send_data['op'] == 'subscribe'
    assert set(send_data['args']) == set(['tickers.BTCUSDT', 'kline.1.BTCUSDT'])
    
    # Test with existing connection
    existing_ws = AsyncMock()
    existing_ws.send = AsyncMock()
    existing_ws.close = AsyncMock()
    
    websocket = await connector._connect_bybit_websocket(
        existing_ws, ['BTCUSDT'], ['ticker']
    )
    
    assert websocket is not None
    assert existing_ws.send.called  # Should send unsubscribe
    assert existing_ws.close.called  # Should close existing connection
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector._connect_bybit_websocket test passed")


@pytest.mark.asyncio
async def test_process_websocket_message_raw(mock_system_config, mock_secret_manager):
    """Test _process_websocket_message_raw method."""
    logger.info("Testing BybitConnector._process_websocket_message_raw")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock _process_websocket_message
    connector._process_websocket_message = AsyncMock()
    
    # Test valid JSON message
    valid_message = json.dumps({
        'topic': 'tickers.BTCUSDT',
        'data': {
            'symbol': 'BTCUSDT',
            'lastPrice': '50000'
        }
    })
    
    await connector._process_websocket_message_raw(valid_message)
    
    assert connector._process_websocket_message.called
    
    # Test invalid JSON message
    connector._process_websocket_message.reset_mock()
    
    with patch.object(connector.logger, 'error') as mock_error:
        await connector._process_websocket_message_raw("invalid json")
        assert mock_error.called
    
    assert not connector._process_websocket_message.called
    
    # Test heartbeat messages
    connector._process_websocket_message.reset_mock()
    
    ping_message = json.dumps({'op': 'ping'})
    pong_message = json.dumps({'op': 'pong'})
    subscribe_message = json.dumps({'op': 'subscribe', 'success': True})
    
    await connector._process_websocket_message_raw(ping_message)
    await connector._process_websocket_message_raw(pong_message)
    await connector._process_websocket_message_raw(subscribe_message)
    
    assert not connector._process_websocket_message.called
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector._process_websocket_message_raw test passed")


@pytest.mark.asyncio
async def test_fetch_funding_rate(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test fetch_funding_rate method."""
    logger.info("Testing BybitConnector.fetch_funding_rate")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock API call
    get_funding_prevrate = AsyncMock(return_value=mock_exchange_data['bybit']['funding_rate'])
    connector.async_exchange.publicGetV5MarketFundingPrevrate = get_funding_prevrate
    
    # Test for linear futures mode
    funding_rate = await connector.fetch_funding_rate('BTC/USDT')
    
    assert funding_rate is not None
    assert funding_rate['symbol'] == 'BTC/USDT'
    assert 'fundingRate' in funding_rate
    assert get_funding_prevrate.called
    assert get_funding_prevrate.call_args[0][0] == {'category': 'linear', 'symbol': 'BTCUSDT'}
    
    # Test for spot mode (should raise error)
    connector.category = 'spot'
    
    with pytest.raises(ValueError):
        await connector.fetch_funding_rate('BTC/USDT')
    
    # Test error response
    connector.category = 'linear'
    get_funding_prevrate.return_value = {'retCode': 10001, 'retMsg': 'Error message'}
    
    with pytest.raises(Exception):
        await connector.fetch_funding_rate('BTC/USDT')
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.fetch_funding_rate test passed")


@pytest.mark.asyncio
async def test_fetch_leverage(patch_ccxt_exchanges, mock_system_config, mock_secret_manager, mock_exchange_data):
    """Test fetch_leverage method."""
    logger.info("Testing BybitConnector.fetch_leverage")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock API call
    get_position_list = AsyncMock(return_value=mock_exchange_data['bybit']['position_info'])
    connector.async_exchange.privateGetV5PositionList = get_position_list
    
    # Test for linear futures mode
    leverage = await connector.fetch_leverage('BTC/USDT')
    
    assert leverage is not None
    assert leverage['symbol'] == 'BTC/USDT'
    assert 'leverage' in leverage
    assert 'marginType' in leverage
    assert get_position_list.called
    assert get_position_list.call_args[0][0] == {'category': 'linear', 'symbol': 'BTCUSDT'}
    
    # Test for spot mode (should raise error)
    connector.category = 'spot'
    
    with pytest.raises(ValueError):
        await connector.fetch_leverage('BTC/USDT')
    
    # Test with no API key
    connector.category = 'linear'
    connector.api_key = None
    
    with pytest.raises(ValueError):
        await connector.fetch_leverage('BTC/USDT')
    
    # Test error response
    connector.api_key = 'test_api_key'
    get_position_list.return_value = {'retCode': 10001, 'retMsg': 'Error message'}
    
    with pytest.raises(Exception):
        await connector.fetch_leverage('BTC/USDT')
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.fetch_leverage test passed")


@pytest.mark.asyncio
async def test_set_leverage(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test set_leverage method."""
    logger.info("Testing BybitConnector.set_leverage")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock API call
    post_leverage_save = AsyncMock(return_value={'retCode': 0, 'retMsg': 'OK'})
    connector.async_exchange.privatePostV5PositionLeverageSave = post_leverage_save
    
    # Test for linear futures mode
    result = await connector.set_leverage('BTC/USDT', 10, 'ISOLATED')
    
    assert result is not None
    assert result['success'] is True
    assert result['symbol'] == 'BTC/USDT'
    assert result['leverage'] == 10
    assert result['marginMode'] == 'ISOLATED'
    assert post_leverage_save.called
    assert post_leverage_save.call_args[0][0] == {
        'category': 'linear',
        'symbol': 'BTCUSDT',
        'leverage': 10,
        'tradeMode': 1
    }
    
    # Test with CROSS margin
    post_leverage_save.reset_mock()
    
    result = await connector.set_leverage('BTC/USDT', 10, 'CROSS')
    
    assert result is not None
    assert post_leverage_save.called
    assert post_leverage_save.call_args[0][0]['tradeMode'] == 0
    
    # Test for spot mode (should raise error)
    connector.category = 'spot'
    
    with pytest.raises(ValueError):
        await connector.set_leverage('BTC/USDT', 10)
    
    # Test with no API key
    connector.category = 'linear'
    connector.api_key = None
    
    with pytest.raises(ValueError):
        await connector.set_leverage('BTC/USDT', 10)
    
    # Test with invalid margin mode
    connector.api_key = 'test_api_key'
    
    with pytest.raises(ValueError):
        await connector.set_leverage('BTC/USDT', 10, 'INVALID')
    
    # Test error response
    post_leverage_save.return_value = {'retCode': 10001, 'retMsg': 'Error message'}
    
    with pytest.raises(Exception):
        await connector.set_leverage('BTC/USDT', 10)
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.set_leverage test passed")


@pytest.mark.asyncio
async def test_fetch_tickers(patch_ccxt_exchanges, mock_system_config, mock_secret_manager):
    """Test fetch_tickers method."""
    logger.info("Testing BybitConnector.fetch_tickers")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret',
        config=mock_system_config,
        secret_manager=mock_secret_manager
    )
    
    # Mock API call for specific symbols
    ticker_response = {
        'retCode': 0,
        'retMsg': 'OK',
        'result': {
            'list': [
                {
                    'symbol': 'BTCUSDT',
                    'lastPrice': '50000',
                    'indexPrice': '49900',
                    'markPrice': '50100',
                    '24hVolume': '1000',
                    '24hTurnover': '50000000'
                }
            ]
        }
    }
    get_tickers = AsyncMock(return_value=ticker_response)
    connector.async_exchange.publicGetV5MarketTickers = get_tickers
    
    # Test for specific symbol
    tickers = await connector.fetch_tickers(['BTC/USDT'])
    
    assert tickers is not None
    assert 'BTC/USDT' in tickers
    assert get_tickers.called
    assert get_tickers.call_args[0][0] == {'category': 'linear', 'symbol': 'BTCUSDT'}
    
    # Mock API call for all symbols
    all_tickers_response = {
        'retCode': 0,
        'retMsg': 'OK',
        'result': {
            'list': [
                {
                    'symbol': 'BTCUSDT',
                    'lastPrice': '50000'
                },
                {
                    'symbol': 'ETHUSDT',
                    'lastPrice': '3000'
                }
            ]
        }
    }
    get_tickers.return_value = all_tickers_response
    get_tickers.reset_mock()
    
    # Test for all symbols
    all_tickers = await connector.fetch_tickers()
    
    assert all_tickers is not None
    assert len(all_tickers) == 2
    assert 'BTC/USDT' in all_tickers
    assert 'ETH/USDT' in all_tickers
    assert get_tickers.called
    assert get_tickers.call_args[0][0] == {'category': 'linear'}
    
    # Test error response
    get_tickers.return_value = {'retCode': 10001, 'retMsg': 'Error message'}
    
    with pytest.raises(Exception):
        await connector.fetch_tickers()
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector.fetch_tickers test passed")


@pytest.mark.asyncio
async def test_parse_bybit_symbol():
    """Test _parse_bybit_symbol method."""
    logger.info("Testing BybitConnector._parse_bybit_symbol")
    
    connector = BybitConnector(
        api_key='test_api_key',
        api_secret='test_api_secret'
    )
    
    # Test common pairs
    assert connector._parse_bybit_symbol('BTCUSDT') == ('BTC', 'USDT')
    assert connector._parse_bybit_symbol('ETHUSDT') == ('ETH', 'USDT')
    assert connector._parse_bybit_symbol('BTCUSD') == ('BTC', 'USD')
    assert connector._parse_bybit_symbol('XRPUSDT') == ('XRP', 'USDT')
    assert connector._parse_bybit_symbol('ETHBTC') == ('ETH', 'BTC')
    
    # Test edge cases
    assert connector._parse_bybit_symbol('DOGEUSDT') == ('DOGE', 'USDT')
    assert connector._parse_bybit_symbol('1INCHUSDT') == ('1INCH', 'USDT')
    
    # Test fallback parsing
    assert connector._parse_bybit_symbol('SOMEUNKNOWN')[0] == 'SOMEUNKN'  # Will use last 4 chars as quote
    
    # Cleanup
    await connector.close()
    
    logger.info("✅ BybitConnector._parse_bybit_symbol test passed")


if __name__ == "__main__":
    pytest.main(["-xvs", __file__])