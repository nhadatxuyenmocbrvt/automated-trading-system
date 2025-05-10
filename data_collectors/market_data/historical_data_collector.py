"""
Thu thập dữ liệu lịch sử từ sàn giao dịch.
File này cung cấp các phương thức để thu thập dữ liệu lịch sử từ các sàn giao dịch,
bao gồm dữ liệu OHLCV (giá mở, cao, thấp, đóng, khối lượng) và tỷ lệ tài trợ (funding rates).
"""

import os
import time
import asyncio
import logging
import inspect
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.constants import Timeframe, ErrorCode
from data_collectors.exchange_api.generic_connector import APIError
from data_collectors.exchange_api.binance_connector import BinanceConnector
from data_collectors.exchange_api.bybit_connector import BybitConnector

class HistoricalDataCollector:
    """
    Lớp thu thập dữ liệu lịch sử từ sàn giao dịch.
    Thu thập dữ liệu OHLCV và tỷ lệ tài trợ từ các sàn giao dịch và quản lý việc lưu trữ chúng.
    """
    
    def __init__(
        self,
        exchange_connector,
        data_dir: Optional[Path] = None,
        logger: Optional[logging.Logger] = None,
        max_workers: int = 4
    ):
        """
        Khởi tạo HistoricalDataCollector.
        
        Args:
            exchange_connector: Đối tượng kết nối với sàn giao dịch
            data_dir: Thư mục lưu dữ liệu
            logger: Logger tùy chỉnh
            max_workers: Số luồng tối đa cho việc thu thập dữ liệu song song
        """
        self.exchange_connector = exchange_connector
        self.exchange_id = exchange_connector.exchange_id
        self.is_futures = hasattr(exchange_connector, 'is_futures') and exchange_connector.is_futures
        
        # Thiết lập logger
        self.logger = logger or get_logger(f"historical_collector_{self.exchange_id}")
        
        # Cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Thư mục lưu dữ liệu
        if data_dir is None:
            base_data_dir = Path(self.system_config.get("data_dir", "data"))
            market_type = "futures" if self.is_futures else "spot"
            self.data_dir = base_data_dir / "collected" / self.exchange_id / market_type
        else:
            self.data_dir = data_dir
        
        # Đảm bảo thư mục tồn tại
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Số luồng tối đa
        self.max_workers = max_workers
        
        # Cache cho metadata của các symbol
        self._symbol_metadata = {}
        
        self.logger.info(f"Đã khởi tạo HistoricalDataCollector cho {self.exchange_id}")
    
    async def collect_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        update_existing: bool = False,
        save_path: Optional[Path] = None,
        save_format: str = 'parquet'
    ) -> pd.DataFrame:
        """
        Thu thập dữ liệu OHLCV cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch (ví dụ: 'BTC/USDT')
            timeframe: Khung thời gian (ví dụ: '1h', '4h', '1d')
            start_time: Thời gian bắt đầu (tùy chọn)
            end_time: Thời gian kết thúc (tùy chọn)
            limit: Số lượng candle tối đa (tùy chọn)
            update_existing: True để cập nhật dữ liệu hiện có, False để ghi đè
            save_path: Đường dẫn file để lưu dữ liệu (tùy chọn)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            
        Returns:
            DataFrame chứa dữ liệu OHLCV
        """
        self.logger.info(f"Thu thập dữ liệu OHLCV cho {symbol} ({timeframe})")
        
        # Kiểm tra cặp giao dịch
        if '/' not in symbol:
            self.logger.warning(f"Cặp giao dịch {symbol} không hợp lệ, thiếu dấu '/'")
            # Thử chuẩn hóa cặp giao dịch
            if self.exchange_id.lower() == "binance":
                if symbol.endswith("USDT") or symbol.endswith("BUSD") or symbol.endswith("USDC"):
                    base = symbol[:-4]
                    quote = symbol[-4:]
                    symbol = f"{base}/{quote}"
                    self.logger.info(f"Đã chuẩn hóa cặp giao dịch thành {symbol}")
        
        # Nếu không cung cấp end_time, sử dụng thời gian hiện tại
        if end_time is None:
            end_time = datetime.now()
        
        # Nếu không cung cấp start_time, lấy dữ liệu 30 ngày trước end_time
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # Chuyển đổi datetime sang timestamp (milliseconds)
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        try:
            # Lấy dữ liệu OHLCV từ sàn giao dịch
            if hasattr(self.exchange_connector, 'fetch_historical_klines'):
                # Sử dụng phương thức đặc biệt nếu có
                ohlcv_data = await self._fetch_historical_klines(
                    symbol, timeframe, start_timestamp, end_timestamp, limit
                )
            else:
                # Sử dụng phương thức chung
                ohlcv_data = await self._fetch_ohlcv(
                    symbol, timeframe, start_timestamp, end_timestamp, limit
                )
            
            # Chuyển đổi dữ liệu thành DataFrame
            df = self._convert_ohlcv_to_dataframe(ohlcv_data, symbol)
            
            if df.empty:
                self.logger.warning(f"Không có dữ liệu OHLCV cho {symbol} ({timeframe})")
                return df
            
            # Lọc dữ liệu theo thời gian
            df = df[(df.index >= start_time) & (df.index <= end_time)]
            
            # Cập nhật dữ liệu hiện có nếu cần
            if update_existing and save_path is not None and save_path.exists():
                df = self._update_existing_data(df, save_path, save_format)
            
            # Lưu dữ liệu nếu cần
            if save_path is not None:
                self._save_dataframe(df, save_path, save_format)
            
            self.logger.info(f"Đã thu thập {len(df)} candles cho {symbol} ({timeframe})")
            return df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu OHLCV cho {symbol}: {str(e)}")
            raise
    
    async def collect_all_ohlcv(
        self,
        symbols: List[str],
        timeframe: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        update_existing: bool = False,
        save_dir: Optional[Path] = None,
        save_format: str = 'parquet',
        concurrency: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu OHLCV cho nhiều cặp giao dịch song song.
        
        Args:
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            start_time: Thời gian bắt đầu (tùy chọn)
            end_time: Thời gian kết thúc (tùy chọn)
            limit: Số lượng candle tối đa (tùy chọn)
            update_existing: True để cập nhật dữ liệu hiện có, False để ghi đè
            save_dir: Thư mục lưu dữ liệu (tùy chọn)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            concurrency: Số tác vụ thực hiện đồng thời (nếu None, sử dụng max_workers)
            
        Returns:
            Dict với key là symbol và value là DataFrame chứa dữ liệu OHLCV
        """
        self.logger.info(f"Thu thập dữ liệu OHLCV cho {len(symbols)} cặp giao dịch ({timeframe})")
        
        # Kiểm tra thư mục lưu dữ liệu
        if save_dir is None:
            save_dir = self.data_dir / timeframe
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo danh sách các tác vụ
        tasks = []
        for symbol in symbols:
            if save_dir:
                symbol_filename = f"{symbol.replace('/', '_')}_{timeframe}.{save_format}"
                save_path = save_dir / symbol_filename
            else:
                save_path = None
            
            task = self.collect_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                update_existing=update_existing,
                save_path=save_path,
                save_format=save_format
            )
            tasks.append(task)
        
        # Thực hiện các tác vụ với giới hạn số lượng tác vụ đồng thời
        results = {}
        chunk_size = concurrency or self.max_workers  # Sử dụng concurrency nếu được cung cấp, nếu không sử dụng max_workers
        
        self.logger.info(f"Thực hiện thu thập với {chunk_size} tác vụ đồng thời")
        
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i+chunk_size]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for j, result in enumerate(chunk_results):
                symbol = symbols[i+j]
                if isinstance(result, Exception):
                    self.logger.error(f"Lỗi khi thu thập dữ liệu cho {symbol}: {str(result)}")
                    results[symbol] = pd.DataFrame()
                else:
                    results[symbol] = result
            
            # Nghỉ một chút để tránh vượt quá giới hạn API
            await asyncio.sleep(1)
        
        return results
    
    async def collect_all_symbols_ohlcv(
        self,
        symbols: Optional[List[str]] = None,
        timeframe: str = '1h',
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        update_existing: bool = False,
        save_dir: Optional[Path] = None,
        save_format: str = 'parquet',
        top_symbols: Optional[int] = None,
        concurrency: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu OHLCV cho tất cả các cặp giao dịch hoặc danh sách cặp được chỉ định.
        
        Args:
            symbols: Danh sách cặp giao dịch (tùy chọn, nếu None sẽ lấy tất cả các cặp có sẵn)
            timeframe: Khung thời gian (mặc định: '1h')
            start_time: Thời gian bắt đầu (tùy chọn)
            end_time: Thời gian kết thúc (tùy chọn)
            limit: Số lượng candle tối đa (tùy chọn)
            update_existing: True để cập nhật dữ liệu hiện có, False để ghi đè
            save_dir: Thư mục lưu dữ liệu (tùy chọn)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            top_symbols: Chỉ lấy n cặp giao dịch hàng đầu (tùy chọn)
            concurrency: Số tác vụ thực hiện đồng thời (nếu None, sử dụng max_workers)
            
        Returns:
            Dict với key là symbol và value là DataFrame chứa dữ liệu OHLCV
        """
        # Nếu không cung cấp danh sách symbols, lấy tất cả các cặp giao dịch có sẵn
        if symbols is None or len(symbols) == 0:
            self.logger.info("Không cung cấp danh sách symbols, lấy danh sách từ sàn giao dịch")
            symbols = await self.get_all_available_symbols()
            
            # Lọc symbols dựa trên các patterns phổ biến
            if self.is_futures:
                # Đối với futures, ưu tiên cặp USDT
                usdt_pairs = [s for s in symbols if s.endswith('/USDT') or s.endswith('/USDT:USDT')]
                if usdt_pairs:
                    symbols = usdt_pairs
            else:
                # Đối với spot, ưu tiên cặp BTC, ETH, USDT
                main_pairs = [
                    s for s in symbols if 
                    s.endswith('/BTC') or 
                    s.endswith('/ETH') or 
                    s.endswith('/USDT') or 
                    s.endswith('/BUSD')
                ]
                if main_pairs:
                    symbols = main_pairs
        
        # Lấy thông tin khối lượng giao dịch để xếp hạng các cặp
        if top_symbols is not None and top_symbols > 0 and len(symbols) > top_symbols:
            self.logger.info(f"Lọc {top_symbols} cặp giao dịch hàng đầu từ {len(symbols)} cặp")
            try:
                # Lấy thông tin tickers để sắp xếp theo khối lượng giao dịch
                tickers = await self._get_tickers_volumes(symbols)
                # Lọc ra top_symbols cặp giao dịch có khối lượng lớn nhất
                volume_sorted_symbols = sorted(
                    tickers.items(), 
                    key=lambda x: x[1].get('quoteVolume', 0) if isinstance(x[1], dict) else 0, 
                    reverse=True
                )
                symbols = [s[0] for s in volume_sorted_symbols[:top_symbols]]
                self.logger.info(f"Đã lọc ra {len(symbols)} cặp giao dịch hàng đầu dựa trên khối lượng")
            except Exception as e:
                self.logger.warning(f"Không thể lọc cặp theo khối lượng: {str(e)}")
                # Nếu không thể sắp xếp theo khối lượng, chỉ lấy top_symbols cặp đầu tiên
                symbols = symbols[:top_symbols]
                self.logger.info(f"Đã lọc ra {len(symbols)} cặp giao dịch đầu tiên")
        
        # Thu thập dữ liệu OHLCV cho các cặp đã lọc
        self.logger.info(f"Thu thập dữ liệu OHLCV cho {len(symbols)} cặp giao dịch")
        return await self.collect_all_ohlcv(
            symbols=symbols,
            timeframe=timeframe,
            start_time=start_time,
            end_time=end_time,
            limit=limit,
            update_existing=update_existing,
            save_dir=save_dir,
            save_format=save_format,
            concurrency=concurrency
        )
    
    async def _get_tickers_volumes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Lấy thông tin khối lượng giao dịch cho các cặp giao dịch.
        
        Args:
            symbols: Danh sách cặp giao dịch
            
        Returns:
            Dict với key là symbol và value là thông tin ticker
        """
        try:
            if hasattr(self.exchange_connector, 'fetch_tickers'):
                # Nếu có phương thức fetch_tickers, sử dụng nó để lấy thông tin tất cả tickers
                tickers = await self.exchange_connector.fetch_tickers(symbols)
                return tickers
            else:
                # Nếu không có, lấy từng ticker một
                tickers = {}
                for symbol in symbols:
                    try:
                        ticker = await self.exchange_connector.fetch_ticker(symbol)
                        tickers[symbol] = ticker
                    except Exception as e:
                        self.logger.debug(f"Không thể lấy ticker cho {symbol}: {str(e)}")
                return tickers
        except Exception as e:
            self.logger.warning(f"Lỗi khi lấy thông tin tickers: {str(e)}")
            return {}
    
    async def collect_funding_rate(
        self,
        symbol: str,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        save_path: Optional[Path] = None,
        save_format: str = 'parquet'
    ) -> pd.DataFrame:
        """
        Thu thập dữ liệu tỷ lệ tài trợ cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch (ví dụ: 'BTC/USDT')
            start_time: Thời gian bắt đầu (tùy chọn)
            end_time: Thời gian kết thúc (tùy chọn)
            limit: Số lượng kết quả tối đa (tùy chọn)
            save_path: Đường dẫn file để lưu dữ liệu (tùy chọn)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            
        Returns:
            DataFrame chứa dữ liệu tỷ lệ tài trợ
        """
        self.logger.info(f"Thu thập dữ liệu tỷ lệ tài trợ cho {symbol}")
        
        # Kiểm tra xem có đang sử dụng thị trường futures không
        if not self.is_futures:
            self.logger.error(f"Không thể thu thập tỷ lệ tài trợ cho thị trường spot")
            return pd.DataFrame()
        
        # Nếu không cung cấp end_time, sử dụng thời gian hiện tại
        if end_time is None:
            end_time = datetime.now()
        
        # Nếu không cung cấp start_time, lấy dữ liệu 30 ngày trước end_time
        if start_time is None:
            start_time = end_time - timedelta(days=30)
        
        # Chuyển đổi datetime sang timestamp (milliseconds)
        start_timestamp = int(start_time.timestamp() * 1000)
        end_timestamp = int(end_time.timestamp() * 1000)
        
        try:
            # Kiểm tra xem connector có hỗ trợ fetch_funding_history không
            if not hasattr(self.exchange_connector, 'fetch_funding_history'):
                self.logger.error(f"Sàn {self.exchange_id} không hỗ trợ thu thập lịch sử tỷ lệ tài trợ")
                return pd.DataFrame()
            
            # Lấy dữ liệu tỷ lệ tài trợ
            funding_data = await self._fetch_funding_history(
                symbol, start_timestamp, end_timestamp, limit
            )
            
            # Chuyển đổi dữ liệu thành DataFrame
            df = self._convert_funding_to_dataframe(funding_data, symbol)
            
            if df.empty:
                self.logger.warning(f"Không có dữ liệu tỷ lệ tài trợ cho {symbol}")
                return df
            
            # Lọc dữ liệu theo thời gian
            df = df[(df.index >= start_time) & (df.index <= end_time)]
            
            # Lưu dữ liệu nếu cần
            if save_path is not None:
                self._save_dataframe(df, save_path, save_format)
            
            self.logger.info(f"Đã thu thập {len(df)} tỷ lệ tài trợ cho {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu tỷ lệ tài trợ cho {symbol}: {str(e)}")
            raise
    
    async def collect_all_funding_rates(
        self,
        symbols: List[str],
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: Optional[int] = None,
        save_dir: Optional[Path] = None,
        save_format: str = 'parquet',
        concurrency: Optional[int] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Thu thập dữ liệu tỷ lệ tài trợ cho nhiều cặp giao dịch song song.
        
        Args:
            symbols: Danh sách cặp giao dịch
            start_time: Thời gian bắt đầu (tùy chọn)
            end_time: Thời gian kết thúc (tùy chọn)
            limit: Số lượng kết quả tối đa (tùy chọn)
            save_dir: Thư mục lưu dữ liệu (tùy chọn)
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            concurrency: Số tác vụ thực hiện đồng thời (nếu None, sử dụng max_workers)
            
        Returns:
            Dict với key là symbol và value là DataFrame chứa dữ liệu tỷ lệ tài trợ
        """
        self.logger.info(f"Thu thập dữ liệu tỷ lệ tài trợ cho {len(symbols)} cặp giao dịch")
        
        # Kiểm tra xem có đang sử dụng thị trường futures không
        if not self.is_futures:
            self.logger.error(f"Không thể thu thập tỷ lệ tài trợ cho thị trường spot")
            return {symbol: pd.DataFrame() for symbol in symbols}
        
        # Kiểm tra thư mục lưu dữ liệu
        if save_dir is None:
            save_dir = self.data_dir / "funding"
        
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo danh sách các tác vụ
        tasks = []
        for symbol in symbols:
            if save_dir:
                symbol_filename = f"{symbol.replace('/', '_')}_funding.{save_format}"
                save_path = save_dir / symbol_filename
            else:
                save_path = None
            
            task = self.collect_funding_rate(
                symbol=symbol,
                start_time=start_time,
                end_time=end_time,
                limit=limit,
                save_path=save_path,
                save_format=save_format
            )
            tasks.append(task)
        
        # Thực hiện các tác vụ với giới hạn số lượng tác vụ đồng thời
        results = {}
        chunk_size = concurrency or self.max_workers  # Sử dụng concurrency nếu được cung cấp, nếu không sử dụng max_workers
        
        self.logger.info(f"Thực hiện thu thập funding rates với {chunk_size} tác vụ đồng thời")
        
        for i in range(0, len(tasks), chunk_size):
            chunk_tasks = tasks[i:i+chunk_size]
            chunk_results = await asyncio.gather(*chunk_tasks, return_exceptions=True)
            
            for j, result in enumerate(chunk_results):
                symbol = symbols[i+j]
                if isinstance(result, Exception):
                    self.logger.error(f"Lỗi khi thu thập dữ liệu tỷ lệ tài trợ cho {symbol}: {str(result)}")
                    results[symbol] = pd.DataFrame()
                else:
                    results[symbol] = result
            
            # Nghỉ một chút để tránh vượt quá giới hạn API
            await asyncio.sleep(1)
        
        return results
    
    async def _fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[List]:
        """
        Lấy dữ liệu OHLCV từ sàn giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            since: Thời gian bắt đầu tính từ millisecond epoch
            until: Thời gian kết thúc tính từ millisecond epoch
            limit: Số lượng candle tối đa
            
        Returns:
            Dữ liệu OHLCV dưới dạng list of lists
        """
        all_candles = []
        current_since = since
        
        while True:
            params = {}
            if until:
                # Một số sàn hỗ trợ tham số endTime/until
                if self.exchange_id.lower() == "binance":
                    params['endTime'] = until
                elif self.exchange_id.lower() == "bybit":
                    params['end_time'] = until
            
            # Lấy dữ liệu OHLCV
            candles = await self.exchange_connector.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=current_since,
                limit=limit,
                params=params
            )
            
            if not candles:
                break
                
            all_candles.extend(candles)
            
            # Cập nhật timestamp cho lần gọi tiếp theo
            last_timestamp = candles[-1][0]
            if last_timestamp <= current_since:
                break
                
            current_since = last_timestamp + 1
            
            # Kiểm tra nếu đã đến hoặc vượt quá thời gian kết thúc
            if until and current_since >= until:
                break
            
            # Kiểm tra nếu đã đủ số lượng candle yêu cầu
            if limit and len(all_candles) >= limit:
                all_candles = all_candles[:limit]
                break
            
            # Tránh rate limit
            await asyncio.sleep(0.5)
        
        return all_candles
    
    async def _fetch_historical_klines(
        self,
        symbol: str,
        timeframe: str,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[List]:
        """
        Lấy dữ liệu lịch sử klines từ sàn giao dịch.
        Sử dụng phương thức đặc biệt nếu có (như fetch_historical_klines của Binance).
        
        Args:
            symbol: Cặp giao dịch
            timeframe: Khung thời gian
            since: Thời gian bắt đầu tính từ millisecond epoch
            until: Thời gian kết thúc tính từ millisecond epoch
            limit: Số lượng candle tối đa
            
        Returns:
            Dữ liệu klines dưới dạng list of lists
        """
        # Sử dụng phương thức fetch_historical_klines nếu có
        if hasattr(self.exchange_connector, 'fetch_historical_klines'):
            # Kiểm tra xem phương thức có phải là coroutine function không
            method = self.exchange_connector.fetch_historical_klines
            
            # Phương thức có thể là một coroutine function hoặc một hàm thông thường
            if inspect.iscoroutinefunction(method):
                # Nếu là coroutine function, await nó
                return await method(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=since,
                    end_time=until,
                    limit=limit
                )
            else:
                # Nếu không phải coroutine function, gọi trực tiếp như một hàm thông thường
                return method(
                    symbol=symbol,
                    interval=timeframe,
                    start_time=since,
                    end_time=until,
                    limit=limit
                )
        # Nếu không có fetch_historical_klines, sử dụng phương thức thông thường
        else:
            return await self._fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since,
                until=until,
                limit=limit
            )
    
    async def _fetch_funding_history(
        self,
        symbol: str,
        since: Optional[int] = None,
        until: Optional[int] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Lấy lịch sử tỷ lệ tài trợ từ sàn giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            since: Thời gian bắt đầu tính từ millisecond epoch
            until: Thời gian kết thúc tính từ millisecond epoch
            limit: Số lượng kết quả tối đa
            
        Returns:
            Lịch sử tỷ lệ tài trợ dưới dạng list of dicts
        """
        all_funding = []
        current_since = since
        
        # Kiểm tra xem connector có hỗ trợ fetch_funding_history không
        if not hasattr(self.exchange_connector, 'fetch_funding_history'):
            self.logger.error(f"Sàn {self.exchange_id} không hỗ trợ thu thập lịch sử tỷ lệ tài trợ")
            return []
        
        # Kiểm tra xem phương thức có phải là coroutine function không
        method = self.exchange_connector.fetch_funding_history
        is_coro = inspect.iscoroutinefunction(method)
        
        while True:
            params = {}
            if until:
                # Một số sàn hỗ trợ tham số endTime/until
                if self.exchange_id.lower() == "binance":
                    params['endTime'] = until
                elif self.exchange_id.lower() == "bybit":
                    params['end_time'] = until
            
            # Lấy dữ liệu tỷ lệ tài trợ
            if is_coro:
                funding_rates = await method(
                    symbol=symbol,
                    since=current_since,
                    limit=limit,
                    params=params
                )
            else:
                funding_rates = method(
                    symbol=symbol,
                    since=current_since,
                    limit=limit,
                    params=params
                )
            
            if not funding_rates:
                break
                
            all_funding.extend(funding_rates)
            
            # Cập nhật timestamp cho lần gọi tiếp theo
            if 'timestamp' in funding_rates[-1]:
                last_timestamp = funding_rates[-1]['timestamp']
            elif 'time' in funding_rates[-1]:
                last_timestamp = funding_rates[-1]['time']
            else:
                self.logger.warning(f"Không thể xác định timestamp trong kết quả tỷ lệ tài trợ")
                break
                
            if last_timestamp <= current_since:
                break
                
            current_since = last_timestamp + 1
            
            # Kiểm tra nếu đã đến hoặc vượt quá thời gian kết thúc
            if until and current_since >= until:
                break
            
            # Kiểm tra nếu đã đủ số lượng kết quả yêu cầu
            if limit and len(all_funding) >= limit:
                all_funding = all_funding[:limit]
                break
            
            # Tránh rate limit
            await asyncio.sleep(0.5)
        
        return all_funding
    
    def _convert_ohlcv_to_dataframe(self, ohlcv_data: List[List], symbol: str) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu OHLCV từ list of lists sang DataFrame.
        
        Args:
            ohlcv_data: Dữ liệu OHLCV dưới dạng list of lists
            symbol: Cặp giao dịch
            
        Returns:
            DataFrame chứa dữ liệu OHLCV
        """
        if not ohlcv_data:
            return pd.DataFrame()
        
        # Tạo DataFrame từ dữ liệu OHLCV
        df = pd.DataFrame(
            ohlcv_data,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        
        # Chuyển đổi timestamp sang datetime index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Thêm thông tin symbol
        df['symbol'] = symbol
        
        # Đảm bảo các cột số có kiểu dữ liệu float
        numeric_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Sắp xếp theo thời gian
        df.sort_index(inplace=True)
        
        return df
    
    def _convert_funding_to_dataframe(self, funding_data: List[Dict], symbol: str) -> pd.DataFrame:
        """
        Chuyển đổi dữ liệu tỷ lệ tài trợ từ list of dicts sang DataFrame.
        
        Args:
            funding_data: Dữ liệu tỷ lệ tài trợ dưới dạng list of dicts
            symbol: Cặp giao dịch
            
        Returns:
            DataFrame chứa dữ liệu tỷ lệ tài trợ
        """
        if not funding_data:
            return pd.DataFrame()
        
        # Tạo DataFrame từ dữ liệu tỷ lệ tài trợ
        df = pd.DataFrame(funding_data)
        
        # Chuẩn hóa tên cột
        column_mapping = {
            'timestamp': 'timestamp',
            'time': 'timestamp',
            'fundingRate': 'funding_rate',
            'funding_rate': 'funding_rate',
            'rate': 'funding_rate',
            'fundingTimestamp': 'funding_timestamp',
            'funding_timestamp': 'funding_timestamp',
        }
        
        df = df.rename(columns={k: v for k, v in column_mapping.items() if k in df.columns})
        
        # Đảm bảo có các cột cần thiết
        if 'timestamp' not in df.columns:
            self.logger.error(f"Không tìm thấy cột 'timestamp' trong dữ liệu tỷ lệ tài trợ")
            return pd.DataFrame()
        
        if 'funding_rate' not in df.columns:
            # Thử tìm cột có tên chứa 'rate'
            rate_columns = [col for col in df.columns if 'rate' in col.lower()]
            if rate_columns:
                df['funding_rate'] = df[rate_columns[0]]
            else:
                self.logger.error(f"Không tìm thấy cột 'funding_rate' trong dữ liệu tỷ lệ tài trợ")
                return pd.DataFrame()
        
        # Chuyển đổi timestamp sang datetime index
        df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('datetime', inplace=True)
        
        # Thêm thông tin symbol
        df['symbol'] = symbol
        
        # Đảm bảo funding_rate có kiểu dữ liệu float
        df['funding_rate'] = pd.to_numeric(df['funding_rate'], errors='coerce')
        
        # Sắp xếp theo thời gian
        df.sort_index(inplace=True)
        
        return df
    
    def _update_existing_data(self, df: pd.DataFrame, file_path: Path, file_format: str) -> pd.DataFrame:
        """
        Cập nhật dữ liệu hiện có với dữ liệu mới.
        
        Args:
            df: DataFrame chứa dữ liệu mới
            file_path: Đường dẫn file dữ liệu hiện có
            file_format: Định dạng file ('parquet', 'csv', 'json')
            
        Returns:
            DataFrame đã được cập nhật
        """
        if not file_path.exists():
            return df
        
        try:
            # Đọc dữ liệu hiện có
            existing_df = None
            
            if file_format == 'parquet':
                existing_df = pd.read_parquet(file_path)
            elif file_format == 'csv':
                existing_df = pd.read_csv(file_path, index_col='datetime', parse_dates=True)
            elif file_format == 'json':
                existing_df = pd.read_json(file_path, orient='records')
                existing_df['datetime'] = pd.to_datetime(existing_df['datetime'])
                existing_df.set_index('datetime', inplace=True)
            
            if existing_df is None or existing_df.empty:
                return df
            
            # Kết hợp dữ liệu cũ và mới
            combined_df = pd.concat([existing_df, df])
            
            # Loại bỏ dữ liệu trùng lặp
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            
            # Sắp xếp theo thời gian
            combined_df.sort_index(inplace=True)
            
            return combined_df
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật dữ liệu hiện có: {str(e)}")
            return df
    
    def _save_dataframe(self, df: pd.DataFrame, file_path: Path, file_format: str) -> bool:
        """
        Lưu DataFrame vào file.
        
        Args:
            df: DataFrame cần lưu
            file_path: Đường dẫn file
            file_format: Định dạng file ('parquet', 'csv', 'json')
            
        Returns:
            True nếu thành công, False nếu thất bại
        """
        try:
            # Đảm bảo thư mục cha tồn tại
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Lưu DataFrame theo định dạng yêu cầu
            if file_format == 'parquet':
                df.to_parquet(file_path, index=True)
            elif file_format == 'csv':
                df.to_csv(file_path, index=True)
            elif file_format == 'json':
                df.reset_index().to_json(file_path, orient='records', date_format='iso')
            else:
                self.logger.error(f"Định dạng file {file_format} không được hỗ trợ")
                return False
            
            self.logger.info(f"Đã lưu dữ liệu vào {file_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu dữ liệu: {str(e)}")
            return False
    
    async def get_symbol_metadata(self, symbol: str) -> Dict:
        """
        Lấy metadata cho một cặp giao dịch.
        
        Args:
            symbol: Cặp giao dịch
            
        Returns:
            Dict chứa thông tin metadata
        """
        # Kiểm tra cache
        if symbol in self._symbol_metadata:
            return self._symbol_metadata[symbol]
        
        try:
            # Lấy thông tin market
            markets = await self.exchange_connector.fetch_markets(force_update=False)
            
            for market in markets:
                if market.get('symbol') == symbol:
                    # Lưu vào cache
                    self._symbol_metadata[symbol] = market
                    return market
            
            self.logger.warning(f"Không tìm thấy thông tin metadata cho {symbol}")
            return {}
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy metadata cho {symbol}: {str(e)}")
            return {}
    
    async def get_all_available_symbols(self) -> List[str]:
        """
        Lấy danh sách tất cả các cặp giao dịch có sẵn.
        
        Returns:
            Danh sách các cặp giao dịch
        """
        try:
            # Lấy thông tin market
            markets = await self.exchange_connector.fetch_markets(force_update=False)
            
            # Trích xuất danh sách symbol
            symbols = [market.get('symbol') for market in markets if 'symbol' in market]
            
            return symbols
            
        except Exception as e:
            self.logger.error(f"Lỗi khi lấy danh sách symbol: {str(e)}")
            return []

async def create_data_collector(
    exchange_id: str,
    api_key: str = '',
    api_secret: str = '',
    testnet: bool = False,
    is_futures: bool = False,
    data_dir: Optional[Path] = None,
    logger: Optional[logging.Logger] = None,
    max_workers: int = 4
) -> HistoricalDataCollector:
    """
    Tạo một phiên bản của HistoricalDataCollector.
    
    Args:
        exchange_id: ID của sàn giao dịch
        api_key: API key (tùy chọn)
        api_secret: API secret (tùy chọn)
        testnet: True để sử dụng testnet
        is_futures: True để sử dụng thị trường futures
        data_dir: Thư mục lưu dữ liệu
        logger: Logger tùy chỉnh
        max_workers: Số luồng tối đa
        
    Returns:
        Phiên bản của HistoricalDataCollector
    """
    # Tạo logger nếu không được cung cấp
    if logger is None:
        logger = get_logger(f"historical_collector_{exchange_id}")
    
    try:
        # Tạo exchange connector tương ứng
        exchange_connector = None
        
        if exchange_id.lower() == "binance":
            # Tạo Binance connector
            exchange_connector = BinanceConnector(
                api_key=api_key,
                api_secret=api_secret,
                is_futures=is_futures,
                testnet=testnet
            )
        elif exchange_id.lower() == "bybit":
            # Tạo Bybit connector
            market_type = "linear" if is_futures else "spot"
            exchange_connector = BybitConnector(
                api_key=api_key,
                api_secret=api_secret,
                market_type=market_type,
                testnet=testnet
            )
        else:
            # Tạo generic connector thông qua ccxt
            from data_collectors.exchange_api.generic_connector import ExchangeConnector
            
            class GenericConnector(ExchangeConnector):
                def _init_ccxt(self):
                    import ccxt
                    params = {
                        'apiKey': self.api_key,
                        'secret': self.api_secret,
                        'timeout': self.timeout,
                        'enableRateLimit': True
                    }
                    
                    if self.testnet:
                        # Thiết lập testnet nếu có
                        params['test'] = True
                    
                    # Tạo đối tượng ccxt exchange
                    return getattr(ccxt, exchange_id)(params)
                
                def _init_mapping(self):
                    # Khởi tạo mapping mặc định
                    self._timeframe_map = {}
                    self._order_type_map = {}
                    self._time_in_force_map = {}
            
            exchange_connector = GenericConnector(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                testnet=testnet
            )
        
        # Khởi tạo kết nối
        await exchange_connector.initialize()
        
        # Tạo HistoricalDataCollector
        collector = HistoricalDataCollector(
            exchange_connector=exchange_connector,
            data_dir=data_dir,
            logger=logger,
            max_workers=max_workers
        )
        
        return collector
        
    except Exception as e:
        logger.error(f"Lỗi khi tạo data collector cho {exchange_id}: {str(e)}")
        raise