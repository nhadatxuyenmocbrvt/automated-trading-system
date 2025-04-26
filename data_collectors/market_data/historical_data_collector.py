"""
Thu thập dữ liệu lịch sử từ các sàn giao dịch.
File này cung cấp các lớp và phương thức để tải và lưu trữ dữ liệu lịch sử
từ các sàn giao dịch tiền điện tử, hỗ trợ nhiều loại dữ liệu khác nhau như
OHLCV, orderbook snapshot, và giao dịch lịch sử.
"""

import os
import time
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
import logging
import json
from pathlib import Path
import concurrent.futures
from functools import partial

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from data_collectors.exchange_api.generic_connector import ExchangeConnector
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.bybit_connector import BybitConnector
from config.logging_config import setup_logger
from config.constants import Timeframe, TIMEFRAME_TO_SECONDS, Exchange, ErrorCode
from config.env import get_env
from config.system_config import DATA_DIR, BASE_DIR

class HistoricalDataCollector:
    """
    Lớp chính để thu thập dữ liệu lịch sử từ các sàn giao dịch.
    Hỗ trợ thu thập đa dạng loại dữ liệu và lưu trữ dưới nhiều định dạng.
    """
    
    def __init__(
        self,
        exchange_connector: ExchangeConnector,
        data_dir: Path = None,
        max_workers: int = 4,
        rate_limit_factor: float = 0.8
    ):
        """
        Khởi tạo bộ thu thập dữ liệu lịch sử.
        
        Args:
            exchange_connector: Kết nối với sàn giao dịch
            data_dir: Thư mục lưu trữ dữ liệu (mặc định là DATA_DIR từ config)
            max_workers: Số luồng tối đa cho việc thu thập song song
            rate_limit_factor: Hệ số để giảm tốc độ gọi API (0.0 - 1.0)
        """
        self.exchange_connector = exchange_connector
        self.exchange_id = exchange_connector.exchange_id
        self.logger = setup_logger(f"historical_data_collector_{self.exchange_id}")
        
        # Thiết lập thư mục lưu trữ dữ liệu
        if data_dir is None:
            self.data_dir = DATA_DIR / 'historical' / self.exchange_id
        else:
            self.data_dir = data_dir / 'historical' / self.exchange_id
        
        # Tạo thư mục lưu trữ nếu chưa tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Thư mục cho từng loại dữ liệu
        self.ohlcv_dir = self.data_dir / 'ohlcv'
        self.orderbook_dir = self.data_dir / 'orderbook'
        self.trades_dir = self.data_dir / 'trades'
        self.funding_dir = self.data_dir / 'funding'
        
        for dir_path in [self.ohlcv_dir, self.orderbook_dir, self.trades_dir, self.funding_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Cấu hình rate limit
        self.rate_limit = exchange_connector.exchange.rateLimit / 1000  # Chuyển ms thành giây
        self.rate_limit_sleep = self.rate_limit * rate_limit_factor
        
        # Cấu hình cho việc thu thập song song
        self.max_workers = max_workers
        self.semaphore = asyncio.Semaphore(max_workers)
        
        self.logger.info(f"Đã khởi tạo HistoricalDataCollector cho {self.exchange_id}")
    
    async def collect_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        save_format: str = 'parquet',
        update_existing: bool = True
    ) -> pd.DataFrame:
        """
        Thu thập dữ liệu OHLCV (Open, High, Low, Close, Volume) cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            timeframe: Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)
            start_time: Thời gian bắt đầu (mặc định là 30 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            limit: Số lượng nến tối đa mỗi lần gọi API
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            update_existing: Cập nhật dữ liệu hiện có nếu có
            
        Returns:
            DataFrame chứa dữ liệu OHLCV
        """
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 30 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=30)
        
        self.logger.info(f"Thu thập OHLCV cho {symbol} ({timeframe}) từ {start_time} đến {end_time}")
        
        # Kiểm tra xem cặp giao dịch có hiệu lực không
        try:
            if symbol not in self.exchange_connector.markets:
                await self.exchange_connector.load_markets(reload=True)
                if symbol not in self.exchange_connector.markets:
                    self.logger.error(f"Cặp giao dịch {symbol} không hợp lệ cho {self.exchange_id}")
                    return pd.DataFrame()
        except Exception as e:
            self.logger.error(f"Lỗi khi kiểm tra cặp giao dịch {symbol}: {e}")
            return pd.DataFrame()
        
        # Xác định đường dẫn file và kiểm tra dữ liệu hiện có
        filename = f"{symbol.replace('/', '_')}_{timeframe}".lower()
        file_path = None
        
        if save_format == 'parquet':
            file_path = self.ohlcv_dir / f"{filename}.parquet"
        elif save_format == 'csv':
            file_path = self.ohlcv_dir / f"{filename}.csv"
        elif save_format == 'json':
            file_path = self.ohlcv_dir / f"{filename}.json"
        else:
            self.logger.warning(f"Định dạng lưu trữ {save_format} không được hỗ trợ, sử dụng parquet")
            file_path = self.ohlcv_dir / f"{filename}.parquet"
        
        # Kiểm tra và tải dữ liệu hiện có
        existing_data = None
        last_timestamp = None
        
        if update_existing and file_path.exists():
            try:
                if save_format == 'parquet':
                    existing_data = pd.read_parquet(file_path)
                elif save_format == 'csv':
                    existing_data = pd.read_csv(file_path, parse_dates=['timestamp'])
                elif save_format == 'json':
                    existing_data = pd.read_json(file_path, orient='records')
                
                if not existing_data.empty:
                    existing_data = existing_data.sort_values('timestamp')
                    last_timestamp = existing_data['timestamp'].max()
                    
                    # Cập nhật thời gian bắt đầu nếu có dữ liệu hiện có
                    if last_timestamp:
                        actual_last_time = pd.to_datetime(last_timestamp).to_pydatetime()
                        # Trừ đi một khoảng thời gian của timeframe để chắc chắn không bỏ lỡ dữ liệu
                        tf_seconds = TIMEFRAME_TO_SECONDS.get(timeframe, 3600)
                        start_time = max(start_time, actual_last_time - timedelta(seconds=tf_seconds))
                        self.logger.info(f"Cập nhật dữ liệu từ {start_time} (dữ liệu cuối {actual_last_time})")
            except Exception as e:
                self.logger.warning(f"Không thể đọc dữ liệu hiện có từ {file_path}: {e}")
                existing_data = None
        
        # Chuyển đổi datetime sang timestamp (ms)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Thu thập dữ liệu
        all_candles = []
        current_start_ts = start_ts
        
        while current_start_ts < end_ts:
            try:
                # Gọi API với giới hạn rate
                async with self.semaphore:
                    candles = await self.exchange_connector.fetch_ohlcv(
                        symbol, timeframe, current_start_ts, limit
                    )
                    await asyncio.sleep(self.rate_limit_sleep)
                
                if not candles or len(candles) == 0:
                    self.logger.debug(f"Không có dữ liệu mới từ {datetime.fromtimestamp(current_start_ts/1000)}")
                    break
                
                all_candles.extend(candles)
                self.logger.debug(f"Đã lấy {len(candles)} nến từ {datetime.fromtimestamp(current_start_ts/1000)}")
                
                # Cập nhật timestamp cho lần gọi tiếp theo
                last_candle_time = candles[-1][0]
                if last_candle_time <= current_start_ts:
                    # Tránh lặp vô hạn nếu không tăng timestamp
                    self.logger.warning("Timestamp không tăng, dừng thu thập")
                    break
                
                current_start_ts = last_candle_time + 1
                
                # Tạm dừng để tránh rate limit
                await asyncio.sleep(self.rate_limit_sleep)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi thu thập OHLCV cho {symbol}: {e}")
                # Tạm dừng dài hơn khi có lỗi
                await asyncio.sleep(self.rate_limit_sleep * 2)
                break
            
            # Kiểm tra đã đạt đến end_time chưa
            if current_start_ts >= end_ts:
                break
        
        # Chuyển đổi dữ liệu thành DataFrame
        if all_candles:
            df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Loại bỏ các bản ghi trùng lặp
            df = df.drop_duplicates(subset=['timestamp'])
            
            # Ghép với dữ liệu hiện có nếu có
            if existing_data is not None and not existing_data.empty:
                # Kết hợp dữ liệu cũ và mới
                combined_df = pd.concat([existing_data, df])
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp')
                df = combined_df
            
            # Lưu dữ liệu
            try:
                if save_format == 'parquet':
                    df.to_parquet(file_path, index=False)
                elif save_format == 'csv':
                    df.to_csv(file_path, index=False)
                elif save_format == 'json':
                    df.to_json(file_path, orient='records', date_format='iso')
                
                self.logger.info(f"Đã lưu {len(df)} bản ghi OHLCV cho {symbol} vào {file_path}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu dữ liệu: {e}")
            
            return df
        else:
            self.logger.warning(f"Không có dữ liệu mới cho {symbol}")
            if existing_data is not None:
                return existing_data
            return pd.DataFrame()
    
    async def collect_orderbook_snapshots(
        self,
        symbol: str,
        interval: int = 3600,  # 1 giờ
        depth: int = 20,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_format: str = 'parquet'
    ) -> List[Dict]:
        """
        Thu thập snapshot của orderbook theo định kỳ.
        
        Args:
            symbol: Cặp giao dịch
            interval: Khoảng thời gian giữa các snapshot (giây)
            depth: Độ sâu của orderbook
            start_time: Thời gian bắt đầu (mặc định là 7 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            
        Returns:
            Danh sách snapshot orderbook
        """
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 7 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=7)
        
        self.logger.info(f"Thu thập orderbook snapshot cho {symbol} từ {start_time} đến {end_time}")
        
        # Xác định đường dẫn file
        filename = f"{symbol.replace('/', '_')}_depth{depth}_interval{interval}s".lower()
        file_path = None
        
        if save_format == 'parquet':
            file_path = self.orderbook_dir / f"{filename}.parquet"
        elif save_format == 'csv':
            file_path = self.orderbook_dir / f"{filename}.csv"
        elif save_format == 'json':
            file_path = self.orderbook_dir / f"{filename}.json"
        else:
            self.logger.warning(f"Định dạng lưu trữ {save_format} không được hỗ trợ, sử dụng parquet")
            file_path = self.orderbook_dir / f"{filename}.parquet"
        
        # Tạo danh sách các mốc thời gian cần lấy snapshot
        time_points = []
        current_time = start_time
        while current_time <= end_time:
            time_points.append(current_time)
            current_time += timedelta(seconds=interval)
        
        if not time_points:
            self.logger.warning("Không có điểm thời gian nào để thu thập")
            return []
        
        # Dữ liệu hiện có
        existing_snapshots = []
        existing_timestamps = set()
        
        if file_path.exists():
            try:
                if save_format == 'parquet':
                    existing_df = pd.read_parquet(file_path)
                elif save_format == 'csv':
                    existing_df = pd.read_csv(file_path)
                    # Chuyển timestamp thành datetime
                    existing_df['timestamp'] = pd.to_datetime(existing_df['timestamp'])
                elif save_format == 'json':
                    with open(file_path, 'r') as f:
                        existing_snapshots = json.load(f)
                    # Chuyển chuỗi timestamp thành datetime object
                    for snapshot in existing_snapshots:
                        timestamp = datetime.fromisoformat(snapshot['timestamp'].replace('Z', '+00:00'))
                        existing_timestamps.add(timestamp.timestamp())
                
                if save_format != 'json':
                    # Chuyển từ dataframe sang list và lấy timestamps
                    existing_snapshots = existing_df.to_dict('records')
                    existing_timestamps = set(pd.to_datetime(existing_df['timestamp']).map(lambda x: x.timestamp()))
                
                self.logger.info(f"Đã tải {len(existing_snapshots)} snapshot hiện có")
            except Exception as e:
                self.logger.warning(f"Không thể đọc dữ liệu hiện có: {e}")
        
        # Thu thập orderbook snapshots
        snapshots = []
        
        for time_point in time_points:
            # Bỏ qua các điểm thời gian đã có dữ liệu
            if time_point.timestamp() in existing_timestamps:
                continue
            
            try:
                # Lấy snapshot orderbook
                async with self.semaphore:
                    orderbook = await self.exchange_connector.fetch_order_book(
                        symbol, depth
                    )
                    await asyncio.sleep(self.rate_limit_sleep)
                
                if orderbook:
                    # Tạo snapshot với thông tin timestamp
                    snapshot = {
                        "symbol": symbol,
                        "timestamp": time_point.isoformat(),
                        "bids": orderbook['bids'],
                        "asks": orderbook['asks'],
                        "datetime": orderbook.get('datetime', None),
                        "nonce": orderbook.get('nonce', None)
                    }
                    snapshots.append(snapshot)
                    self.logger.debug(f"Đã lấy snapshot cho {symbol} tại {time_point}")
            
            except Exception as e:
                self.logger.error(f"Lỗi khi lấy orderbook cho {symbol} tại {time_point}: {e}")
            
            # Tạm dừng để tránh rate limit
            await asyncio.sleep(self.rate_limit_sleep)
        
        # Kết hợp với dữ liệu hiện có
        all_snapshots = existing_snapshots + snapshots
        
        # Sắp xếp theo thời gian
        all_snapshots.sort(key=lambda x: x['timestamp'])
        
        # Lưu dữ liệu
        if all_snapshots:
            try:
                if save_format == 'parquet':
                    pd.DataFrame(all_snapshots).to_parquet(file_path, index=False)
                elif save_format == 'csv':
                    pd.DataFrame(all_snapshots).to_csv(file_path, index=False)
                elif save_format == 'json':
                    with open(file_path, 'w') as f:
                        json.dump(all_snapshots, f, indent=2)
                
                self.logger.info(f"Đã lưu {len(all_snapshots)} orderbook snapshot cho {symbol}")
            
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu dữ liệu orderbook: {e}")
        
        return all_snapshots
    
    async def collect_historical_trades(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 1000,
        save_format: str = 'parquet',
        update_existing: bool = True
    ) -> pd.DataFrame:
        """
        Thu thập lịch sử giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            start_time: Thời gian bắt đầu (mặc định là 1 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            limit: Số lượng giao dịch tối đa mỗi lần gọi API
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            update_existing: Cập nhật dữ liệu hiện có nếu có
            
        Returns:
            DataFrame chứa lịch sử giao dịch
        """
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 1 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=1)
        
        self.logger.info(f"Thu thập lịch sử giao dịch cho {symbol} từ {start_time} đến {end_time}")
        
        # Xác định đường dẫn file
        filename = f"{symbol.replace('/', '_')}_trades".lower()
        file_path = None
        
        if save_format == 'parquet':
            file_path = self.trades_dir / f"{filename}.parquet"
        elif save_format == 'csv':
            file_path = self.trades_dir / f"{filename}.csv"
        elif save_format == 'json':
            file_path = self.trades_dir / f"{filename}.json"
        else:
            self.logger.warning(f"Định dạng lưu trữ {save_format} không được hỗ trợ, sử dụng parquet")
            file_path = self.trades_dir / f"{filename}.parquet"
        
        # Kiểm tra và tải dữ liệu hiện có
        existing_data = None
        last_timestamp = None
        
        if update_existing and file_path.exists():
            try:
                if save_format == 'parquet':
                    existing_data = pd.read_parquet(file_path)
                elif save_format == 'csv':
                    existing_data = pd.read_csv(file_path, parse_dates=['timestamp'])
                elif save_format == 'json':
                    existing_data = pd.read_json(file_path, orient='records')
                
                if not existing_data.empty:
                    existing_data = existing_data.sort_values('timestamp')
                    last_timestamp = existing_data['timestamp'].max()
                    
                    # Cập nhật thời gian bắt đầu nếu có dữ liệu hiện có
                    if last_timestamp:
                        actual_last_time = pd.to_datetime(last_timestamp).to_pydatetime()
                        # Bắt đầu từ thời điểm cuối cùng có dữ liệu
                        start_time = max(start_time, actual_last_time)
                        self.logger.info(f"Cập nhật dữ liệu từ {start_time} (dữ liệu cuối {actual_last_time})")
            except Exception as e:
                self.logger.warning(f"Không thể đọc dữ liệu hiện có từ {file_path}: {e}")
                existing_data = None
        
        # Chuyển đổi datetime sang timestamp (ms)
        start_ts = int(start_time.timestamp() * 1000)
        end_ts = int(end_time.timestamp() * 1000)
        
        # Thu thập dữ liệu
        all_trades = []
        current_since = start_ts
        
        while current_since < end_ts:
            try:
                # Gọi API với giới hạn rate
                async with self.semaphore:
                    trades = await self.exchange_connector.fetch_trades(
                        symbol, current_since, limit
                    )
                    await asyncio.sleep(self.rate_limit_sleep)
                
                if not trades or len(trades) == 0:
                    self.logger.debug(f"Không có giao dịch mới từ {datetime.fromtimestamp(current_since/1000)}")
                    break
                
                all_trades.extend(trades)
                self.logger.debug(f"Đã lấy {len(trades)} giao dịch từ {datetime.fromtimestamp(current_since/1000)}")
                
                # Cập nhật timestamp cho lần gọi tiếp theo
                last_trade_time = max(trade['timestamp'] for trade in trades)
                if last_trade_time <= current_since:
                    # Tránh lặp vô hạn nếu không tăng timestamp
                    self.logger.warning("Timestamp không tăng, dừng thu thập")
                    break
                
                current_since = last_trade_time + 1
                
                # Tạm dừng để tránh rate limit
                await asyncio.sleep(self.rate_limit_sleep)
                
            except Exception as e:
                self.logger.error(f"Lỗi khi thu thập lịch sử giao dịch cho {symbol}: {e}")
                # Tạm dừng dài hơn khi có lỗi
                await asyncio.sleep(self.rate_limit_sleep * 2)
                break
            
            # Kiểm tra đã đạt đến end_time chưa
            if current_since >= end_ts:
                break
        
        # Chuyển đổi dữ liệu thành DataFrame
        if all_trades:
            # Chuẩn hóa dữ liệu
            normalized_trades = []
            
            for trade in all_trades:
                normalized_trade = {
                    'id': trade.get('id', None),
                    'timestamp': trade.get('timestamp', None),
                    'datetime': trade.get('datetime', None),
                    'symbol': trade.get('symbol', symbol),
                    'side': trade.get('side', None),
                    'price': trade.get('price', None),
                    'amount': trade.get('amount', None),
                    'cost': trade.get('cost', None),
                    'fee': trade.get('fee', None),
                    'fee_currency': trade.get('feeCurrency', None) if 'feeCurrency' in trade else None,
                    'type': trade.get('type', None),
                    'takerOrMaker': trade.get('takerOrMaker', None),
                }
                normalized_trades.append(normalized_trade)
            
            df = pd.DataFrame(normalized_trades)
            
            # Chuyển đổi timestamp sang datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            
            # Loại bỏ các bản ghi trùng lặp
            if 'id' in df.columns:
                df = df.drop_duplicates(subset=['id'])
            else:
                # Nếu không có id, dùng timestamp và price để xác định trùng lặp
                df = df.drop_duplicates(subset=['timestamp', 'price', 'amount'])
            
            # Ghép với dữ liệu hiện có nếu có
            if existing_data is not None and not existing_data.empty:
                # Kết hợp dữ liệu cũ và mới
                combined_df = pd.concat([existing_data, df])
                
                # Loại bỏ trùng lặp
                if 'id' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['id'])
                else:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp', 'price', 'amount'])
                
                combined_df = combined_df.sort_values('timestamp')
                df = combined_df
            
            # Lưu dữ liệu
            try:
                if save_format == 'parquet':
                    df.to_parquet(file_path, index=False)
                elif save_format == 'csv':
                    df.to_csv(file_path, index=False)
                elif save_format == 'json':
                    df.to_json(file_path, orient='records', date_format='iso')
                
                self.logger.info(f"Đã lưu {len(df)} bản ghi giao dịch cho {symbol} vào {file_path}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu dữ liệu giao dịch: {e}")
            
            return df
        else:
            self.logger.warning(f"Không có dữ liệu giao dịch mới cho {symbol}")
            if existing_data is not None:
                return existing_data
            return pd.DataFrame()
    
    async def collect_funding_rates(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        save_format: str = 'parquet',
        update_existing: bool = True
    ) -> pd.DataFrame:
        """
        Thu thập tỷ lệ tài trợ cho hợp đồng tương lai.
        
        Args:
            symbol: Cặp giao dịch
            start_time: Thời gian bắt đầu (mặc định là 30 ngày trước)
            end_time: Thời gian kết thúc (mặc định là hiện tại)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            update_existing: Cập nhật dữ liệu hiện có nếu có
            
        Returns:
            DataFrame chứa dữ liệu tỷ lệ tài trợ
        """
        # Xác định thời gian mặc định nếu không được cung cấp
        if end_time is None:
            end_time = datetime.now()
        
        if start_time is None:
            # Mặc định lấy dữ liệu 30 ngày
            start_time = datetime.now().replace(
                hour=0, minute=0, second=0, microsecond=0
            ) - timedelta(days=30)
        
        self.logger.info(f"Thu thập tỷ lệ tài trợ cho {symbol} từ {start_time} đến {end_time}")
        
        # Xác định đường dẫn file
        filename = f"{symbol.replace('/', '_')}_funding".lower()
        file_path = None
        
        if save_format == 'parquet':
            file_path = self.funding_dir / f"{filename}.parquet"
        elif save_format == 'csv':
            file_path = self.funding_dir / f"{filename}.csv"
        elif save_format == 'json':
            file_path = self.funding_dir / f"{filename}.json"
        else:
            self.logger.warning(f"Định dạng lưu trữ {save_format} không được hỗ trợ, sử dụng parquet")
            file_path = self.funding_dir / f"{filename}.parquet"
        
        # Tỷ lệ tài trợ chỉ khả dụng cho hợp đồng tương lai
        if not hasattr(self.exchange_connector, 'fetch_funding_rate'):
            self.logger.error(f"Sàn giao dịch {self.exchange_id} không hỗ trợ thu thập tỷ lệ tài trợ")
            return pd.DataFrame()
        
        # Kiểm tra và tải dữ liệu hiện có
        existing_data = None
        
        if update_existing and file_path.exists():
            try:
                if save_format == 'parquet':
                    existing_data = pd.read_parquet(file_path)
                elif save_format == 'csv':
                    existing_data = pd.read_csv(file_path, parse_dates=['timestamp'])
                elif save_format == 'json':
                    existing_data = pd.read_json(file_path, orient='records')
                
                self.logger.info(f"Đã tải {len(existing_data)} bản ghi tỷ lệ tài trợ hiện có")
            except Exception as e:
                self.logger.warning(f"Không thể đọc dữ liệu hiện có từ {file_path}: {e}")
                existing_data = None
        
        # Thu thập dữ liệu
        try:
            # Gọi API để lấy tỷ lệ tài trợ hiện tại
            async with self.semaphore:
                funding_rate = await self.exchange_connector.fetch_funding_rate(symbol)
                await asyncio.sleep(self.rate_limit_sleep)
            
            if not funding_rate:
                self.logger.warning(f"Không có dữ liệu tỷ lệ tài trợ cho {symbol}")
                return pd.DataFrame() if existing_data is None else existing_data
            
            # Tạo DataFrame
            new_data = pd.DataFrame([{
                'symbol': symbol,
                'timestamp': pd.to_datetime(funding_rate.get('timestamp', datetime.now().timestamp() * 1000), unit='ms'),
                'fundingRate': funding_rate.get('fundingRate', None),
                'fundingTime': pd.to_datetime(funding_rate.get('fundingTime', None), unit='ms') if funding_rate.get('fundingTime') else None,
                'datetime': funding_rate.get('datetime', None)
            }])
            
            # Thêm dữ liệu mới vào dữ liệu hiện có
            if existing_data is not None and not existing_data.empty:
                # Kết hợp dữ liệu
                combined_df = pd.concat([existing_data, new_data])
                # Loại bỏ trùng lặp
                combined_df = combined_df.drop_duplicates(subset=['timestamp'])
                combined_df = combined_df.sort_values('timestamp')
                new_data = combined_df
            
            # Lưu dữ liệu
            try:
                if save_format == 'parquet':
                    new_data.to_parquet(file_path, index=False)
                elif save_format == 'csv':
                    new_data.to_csv(file_path, index=False)
                elif save_format == 'json':
                    new_data.to_json(file_path, orient='records', date_format='iso')
                
                self.logger.info(f"Đã lưu {len(new_data)} bản ghi tỷ lệ tài trợ cho {symbol}")
                
            except Exception as e:
                self.logger.error(f"Lỗi khi lưu dữ liệu tỷ lệ tài trợ: {e}")
            
            return new_data
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập tỷ lệ tài trợ cho {symbol}: {e}")
            if existing_data is not None:
                return existing_data
            return pd.DataFrame()
    
    async def collect_all_symbols_ohlcv(
        self,
        symbols: List[str],
        timeframe: str = '1h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        concurrency: int = 3,
        save_format: str = 'parquet'
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu OHLCV cho nhiều cặp giao dịch.
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            start_time: Thời gian bắt đầu
            end_time: Thời gian kết thúc
            concurrency: Số lượng cặp giao dịch thu thập đồng thời
            save_format: Định dạng lưu trữ
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        self.logger.info(f"Thu thập OHLCV cho {len(symbols)} cặp giao dịch với {timeframe}")
        
        # Giới hạn số lượng concurrency để tránh rate limit
        max_concurrent = min(concurrency, self.max_workers)
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_symbol_data(symbol):
            async with semaphore:
                self.logger.debug(f"Bắt đầu thu thập dữ liệu cho {symbol}")
                try:
                    df = await self.collect_ohlcv(
                        symbol, timeframe, start_time, end_time, save_format=save_format
                    )
                    return symbol, df
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập dữ liệu cho {symbol}: {e}")
                    return symbol, pd.DataFrame()
        
        # Tạo danh sách các task
        tasks = [fetch_symbol_data(symbol) for symbol in symbols]
        
        # Thực hiện các task
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        data_dict = {}
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Lỗi khi thu thập dữ liệu: {res}")
            else:
                symbol, df = res
                data_dict[symbol] = df
        
        self.logger.info(f"Đã thu thập dữ liệu cho {len(data_dict)} cặp giao dịch")
        return data_dict
    
    async def collect_all_funding_rates(
        self,
        symbols: List[str],
        save_format: str = 'parquet'
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập tỷ lệ tài trợ cho nhiều cặp giao dịch.
        
        Args:
            symbols: Danh sách cặp giao dịch
            save_format: Định dạng lưu trữ
            
        Returns:
            Dict với key là symbol và value là DataFrame
        """
        self.logger.info(f"Thu thập tỷ lệ tài trợ cho {len(symbols)} cặp giao dịch")
        
        # Xác định số lượng concurrency tối đa
        max_concurrent = min(5, self.max_workers)  # Giới hạn thấp hơn cho funding rate
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def fetch_funding_data(symbol):
            async with semaphore:
                self.logger.debug(f"Bắt đầu thu thập tỷ lệ tài trợ cho {symbol}")
                try:
                    df = await self.collect_funding_rates(
                        symbol, save_format=save_format
                    )
                    return symbol, df
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập tỷ lệ tài trợ cho {symbol}: {e}")
                    return symbol, pd.DataFrame()
        
        # Tạo danh sách các task
        tasks = [fetch_funding_data(symbol) for symbol in symbols]
        
        # Thực hiện các task
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Xử lý kết quả
        data_dict = {}
        for res in results:
            if isinstance(res, Exception):
                self.logger.error(f"Lỗi khi thu thập tỷ lệ tài trợ: {res}")
            else:
                symbol, df = res
                data_dict[symbol] = df
        
        self.logger.info(f"Đã thu thập tỷ lệ tài trợ cho {len(data_dict)} cặp giao dịch")
        return data_dict
    
    @staticmethod
    def get_available_timeframes() -> Dict[str, int]:
        """
        Lấy danh sách khung thời gian có sẵn.
        
        Returns:
            Dict với key là tên timeframe và value là số giây
        """
        return TIMEFRAME_TO_SECONDS
    
    def get_local_data_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dữ liệu đã được lưu trữ cục bộ.
        
        Returns:
            Dict chứa thông tin về dữ liệu
        """
        info = {
            "exchange": self.exchange_id,
            "ohlcv": {},
            "orderbook": {},
            "trades": {},
            "funding": {}
        }
        
        # Thu thập thông tin về OHLCV
        ohlcv_files = list(self.ohlcv_dir.glob("*.parquet")) + list(self.ohlcv_dir.glob("*.csv")) + list(self.ohlcv_dir.glob("*.json"))
        for file in ohlcv_files:
            try:
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                elif file.suffix == '.csv':
                    df = pd.read_csv(file)
                elif file.suffix == '.json':
                    df = pd.read_json(file)
                
                symbol = file.stem.split('_')[0]
                timeframe = file.stem.split('_')[1] if len(file.stem.split('_')) > 1 else 'unknown'
                
                if 'timestamp' in df.columns:
                    min_date = pd.to_datetime(df['timestamp'].min()).strftime('%Y-%m-%d')
                    max_date = pd.to_datetime(df['timestamp'].max()).strftime('%Y-%m-%d')
                    info["ohlcv"][file.stem] = {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "rows": len(df),
                        "start_date": min_date,
                        "end_date": max_date,
                        "file": str(file)
                    }
            except Exception as e:
                self.logger.warning(f"Không thể đọc file {file}: {e}")
        
        # Thu thập thông tin về orderbook
        orderbook_files = list(self.orderbook_dir.glob("*.parquet")) + list(self.orderbook_dir.glob("*.csv")) + list(self.orderbook_dir.glob("*.json"))
        for file in orderbook_files:
            try:
                info["orderbook"][file.stem] = {
                    "file": str(file),
                    "size": file.stat().st_size
                }
            except Exception as e:
                self.logger.warning(f"Không thể đọc file {file}: {e}")
        
        # Thu thập thông tin về trades
        trades_files = list(self.trades_dir.glob("*.parquet")) + list(self.trades_dir.glob("*.csv")) + list(self.trades_dir.glob("*.json"))
        for file in trades_files:
            try:
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                elif file.suffix == '.csv':
                    df = pd.read_csv(file)
                elif file.suffix == '.json':
                    df = pd.read_json(file)
                
                symbol = file.stem.split('_')[0]
                
                if 'timestamp' in df.columns:
                    min_date = pd.to_datetime(df['timestamp'].min()).strftime('%Y-%m-%d')
                    max_date = pd.to_datetime(df['timestamp'].max()).strftime('%Y-%m-%d')
                    info["trades"][file.stem] = {
                        "symbol": symbol,
                        "rows": len(df),
                        "start_date": min_date,
                        "end_date": max_date,
                        "file": str(file)
                    }
            except Exception as e:
                self.logger.warning(f"Không thể đọc file {file}: {e}")
        
        # Thu thập thông tin về funding
        funding_files = list(self.funding_dir.glob("*.parquet")) + list(self.funding_dir.glob("*.csv")) + list(self.funding_dir.glob("*.json"))
        for file in funding_files:
            try:
                if file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                elif file.suffix == '.csv':
                    df = pd.read_csv(file)
                elif file.suffix == '.json':
                    df = pd.read_json(file)
                
                symbol = file.stem.split('_')[0]
                
                if 'timestamp' in df.columns:
                    min_date = pd.to_datetime(df['timestamp'].min()).strftime('%Y-%m-%d')
                    max_date = pd.to_datetime(df['timestamp'].max()).strftime('%Y-%m-%d')
                    info["funding"][file.stem] = {
                        "symbol": symbol,
                        "rows": len(df),
                        "start_date": min_date,
                        "end_date": max_date,
                        "file": str(file)
                    }
            except Exception as e:
                self.logger.warning(f"Không thể đọc file {file}: {e}")
        
        return info

# Factory function
async def create_data_collector(
    exchange_id: str,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None,
    sandbox: bool = True,
    is_futures: bool = False,
    max_workers: int = 4
) -> HistoricalDataCollector:
    """
    Tạo một instance của HistoricalDataCollector cho sàn giao dịch cụ thể.
    
    Args:
        exchange_id: ID của sàn giao dịch
        api_key: Khóa API
        api_secret: Mật khẩu API
        sandbox: Sử dụng môi trường testnet
        is_futures: Sử dụng tài khoản futures
        max_workers: Số luồng tối đa cho việc thu thập song song
        
    Returns:
        Instance của HistoricalDataCollector
    """
    # Tạo connector cho sàn giao dịch
    exchange_connector = None
    
    if exchange_id.lower() == 'binance':
        exchange_connector = BinanceConnector(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            futures=is_futures
        )
    elif exchange_id.lower() == 'bybit':
        exchange_connector = BybitConnector(
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox,
            category='linear' if is_futures else 'spot'
        )
    else:
        # Sử dụng ExchangeConnector cho các sàn khác
        exchange_connector = ExchangeConnector(
            exchange_id=exchange_id,
            api_key=api_key,
            api_secret=api_secret,
            sandbox=sandbox
        )
    
    # Khởi tạo connector
    await exchange_connector.initialize()
    
    # Tạo collector
    collector = HistoricalDataCollector(
        exchange_connector=exchange_connector,
        max_workers=max_workers
    )
    
    return collector

async def main():
    """
    Hàm chính để chạy collector.
    """
    # Đọc thông tin cấu hình từ biến môi trường
    exchange_id = get_env('DEFAULT_EXCHANGE', 'binance')
    api_key = get_env(f'{exchange_id.upper()}_API_KEY', '')
    api_secret = get_env(f'{exchange_id.upper()}_API_SECRET', '')
    
    # Khởi tạo collector
    collector = await create_data_collector(
        exchange_id=exchange_id,
        api_key=api_key,
        api_secret=api_secret,
        sandbox=True
    )
    
    # Lấy danh sách cặp giao dịch phổ biến
    symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    # Thu thập dữ liệu OHLCV
    for symbol in symbols:
        await collector.collect_ohlcv(
            symbol=symbol,
            timeframe='1h',
            start_time=datetime.now() - timedelta(days=7),
            end_time=datetime.now()
        )
    
    # Thu thập orderbook snapshot
    for symbol in symbols:
        await collector.collect_orderbook_snapshots(
            symbol=symbol,
            interval=3600,  # 1 giờ
            depth=20,
            start_time=datetime.now() - timedelta(days=1),
            end_time=datetime.now()
        )
    
    # Đóng kết nối
    await collector.exchange_connector.close()

if __name__ == "__main__":
    asyncio.run(main())