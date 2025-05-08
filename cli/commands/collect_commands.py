"""
Lệnh thu thập dữ liệu cho hệ thống giao dịch tự động.
File này định nghĩa các lệnh thu thập dữ liệu thị trường từ các sàn giao dịch,
bao gồm dữ liệu lịch sử, dữ liệu thời gian thực và dữ liệu orderbook.
"""

import os
import sys
import argparse
import asyncio
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from pathlib import Path
from datetime import datetime, timedelta

# Thêm thư mục gốc vào path để import các module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

# Import các module cần thiết
from config.logging_config import get_logger
from config.system_config import get_system_config, DATA_DIR
from config.env import get_env
from data_collectors.market_data.historical_data_collector import create_data_collector
from data_collectors.market_data.realtime_data_stream import create_realtime_stream
from data_collectors.market_data.orderbook_collector import create_orderbook_collector

class CollectCommands:
    """
    Lớp quản lý các lệnh thu thập dữ liệu.
    Hỗ trợ thu thập dữ liệu lịch sử, dữ liệu thời gian thực và dữ liệu orderbook.
    """
    
    def __init__(self, system=None):
        """
        Khởi tạo CollectCommands.
        
        Args:
            system: Instance của AutomatedTradingSystem (tùy chọn)
        """
        self.logger = get_logger("collect_commands")
        self.system = system
        self.data_dir = DATA_DIR if system is None else system.data_dir
        
        # Các thành phần thu thập dữ liệu
        self.historical_collector = None
        self.realtime_stream = None
        self.orderbook_collector = None
        
        # Lưu trữ trạng thái thu thập
        self.is_collecting = False
        self.active_tasks = []
        
        self.logger.debug("Đã khởi tạo CollectCommands")
    
    async def collect_historical_data(
        self,
        exchange_id: str,
        symbols: List[str],
        timeframe: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        is_futures: bool = False,
        output_dir: Optional[Path] = None,
        save_format: str = 'parquet',
        force_update: bool = False,
        max_workers: int = 4
    ) -> Dict[str, Path]:
        """
        Thu thập dữ liệu lịch sử.
        
        Args:
            exchange_id: ID của sàn giao dịch
            symbols: Danh sách cặp giao dịch
            timeframe: Khung thời gian
            start_date: Ngày bắt đầu (định dạng YYYY-MM-DD)
            end_date: Ngày kết thúc (định dạng YYYY-MM-DD)
            is_futures: Sử dụng thị trường futures
            output_dir: Thư mục lưu dữ liệu
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            force_update: Cập nhật dữ liệu kể cả khi đã có
            max_workers: Số luồng tối đa
            
        Returns:
            Dict với key là symbol và value là đường dẫn file dữ liệu
        """
        self.logger.info(f"Thu thập dữ liệu lịch sử cho {len(symbols)} cặp giao dịch từ {exchange_id}")
        
        # Kiểm tra và chuyển đổi ngày
        start_time = None
        end_time = None
        
        if start_date:
            start_time = datetime.strptime(start_date, "%Y-%m-%d")
        
        if end_date:
            end_time = datetime.strptime(end_date, "%Y-%m-%d")
        
        # Xác định thư mục đầu ra
        if output_dir is None:
            # Tạo cấu trúc thư mục theo exchange/spot_futures/timeframe
            market_type = "futures" if is_futures else "spot"
            output_dir = self.data_dir / "historical" / exchange_id / market_type / timeframe
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Lấy API key và secret từ biến môi trường
            api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
            api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
            
            # Tạo collector
            self.historical_collector = await create_data_collector(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                testnet=False,
                is_futures=is_futures,
                max_workers=max_workers
            )
            
            # Thu thập dữ liệu cho mỗi symbol
            result_files = {}
            
            for symbol in symbols:
                self.logger.info(f"Đang thu thập {symbol} ({timeframe}) từ {start_date or 'mặc định'} đến {end_date or 'hiện tại'}")
                
                try:
                    # Thu thập dữ liệu OHLCV
                    df = await self.historical_collector.collect_ohlcv(
                        symbol=symbol,
                        timeframe=timeframe,
                        start_time=start_time,
                        end_time=end_time,
                        save_format=save_format,
                        update_existing=not force_update
                    )
                    
                    if df is not None and not df.empty:
                        # Tạo tên file
                        filename = f"{symbol.replace('/', '_')}_{timeframe}".lower()
                        
                        if save_format == 'parquet':
                            file_path = output_dir / f"{filename}.parquet"
                        elif save_format == 'csv':
                            file_path = output_dir / f"{filename}.csv"
                        elif save_format == 'json':
                            file_path = output_dir / f"{filename}.json"
                        else:
                            file_path = output_dir / f"{filename}.parquet"
                        
                        result_files[symbol] = file_path
                        self.logger.info(f"Đã thu thập {len(df)} bản ghi cho {symbol}")
                    else:
                        self.logger.warning(f"Không có dữ liệu cho {symbol}")
                
                except Exception as e:
                    self.logger.error(f"Lỗi khi thu thập dữ liệu cho {symbol}: {str(e)}")
            
            return result_files
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu lịch sử: {str(e)}")
            return {}
        finally:
            # Đóng kết nối
            if self.historical_collector:
                await self.historical_collector.exchange_connector.close()
                self.historical_collector = None
    
    async def collect_realtime_data(
        self,
        exchange_id: str,
        symbols: List[str],
        channels: List[str],
        duration: int = 3600,  # Thời gian thu thập (giây)
        is_futures: bool = False,
        output_dir: Optional[Path] = None
    ) -> bool:
        """
        Thu thập dữ liệu thời gian thực.
        
        Args:
            exchange_id: ID của sàn giao dịch
            symbols: Danh sách cặp giao dịch
            channels: Danh sách kênh dữ liệu ('ticker', 'kline_1m', 'orderbook', 'trade')
            duration: Thời gian thu thập (giây)
            is_futures: Sử dụng thị trường futures
            output_dir: Thư mục lưu dữ liệu
            
        Returns:
            True nếu thành công, False nếu có lỗi
        """
        self.logger.info(f"Thu thập dữ liệu thời gian thực cho {len(symbols)} cặp giao dịch từ {exchange_id}")
        
        # Xác định thư mục đầu ra
        if output_dir is None:
            # Tạo cấu trúc thư mục theo exchange/spot_futures/realtime
            market_type = "futures" if is_futures else "spot"
            output_dir = self.data_dir / "realtime" / exchange_id / market_type
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Lấy API key và secret từ biến môi trường
            api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
            api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
            
            # Tạo các handler cho dữ liệu
            from data_collectors.market_data.realtime_data_stream import ConsoleOutputHandler, CSVStorageHandler
            
            # Handler console để in dữ liệu
            console_handler = ConsoleOutputHandler(name=f"{exchange_id}_console", log_level="INFO")
            
            # Handler CSV để lưu dữ liệu
            csv_handler = CSVStorageHandler(
                name=f"{exchange_id}_csv_storage",
                data_dir=output_dir
            )
            
            # Tạo realtime stream
            self.realtime_stream = await create_realtime_stream(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                sandbox=False,
                is_futures=is_futures,
                data_handlers=[console_handler, csv_handler]
            )
            
            # Bắt đầu stream
            await self.realtime_stream.start()
            
            # Đăng ký nhận dữ liệu cho các cặp giao dịch
            for symbol in symbols:
                await self.realtime_stream.subscribe(symbol, channels)
            
            self.logger.info(f"Đã bắt đầu thu thập dữ liệu thời gian thực cho {len(symbols)} cặp giao dịch")
            self.logger.info(f"Thu thập sẽ diễn ra trong {duration} giây...")
            
            # Đợi trong thời gian xác định
            await asyncio.sleep(duration)
            
            # Dừng thu thập
            await self.realtime_stream.stop()
            
            # Đóng các handler
            await csv_handler.close()
            
            self.logger.info("Đã hoàn thành thu thập dữ liệu thời gian thực")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu thời gian thực: {str(e)}")
            return False
        finally:
            # Đóng kết nối
            if self.realtime_stream:
                try:
                    await self.realtime_stream.stop()
                    await self.realtime_stream.exchange_connector.close()
                except:
                    pass
                self.realtime_stream = None
    
    async def collect_orderbook_data(
        self,
        exchange_id: str,
        symbols: List[str],
        snapshot_interval: int = 60,  # Thời gian giữa các snapshot (giây)
        depth: int = 20,  # Độ sâu của orderbook
        duration: int = 3600,  # Thời gian thu thập (giây)
        is_futures: bool = False,
        output_dir: Optional[Path] = None
    ) -> bool:
        """
        Thu thập dữ liệu orderbook.
        
        Args:
            exchange_id: ID của sàn giao dịch
            symbols: Danh sách cặp giao dịch
            snapshot_interval: Thời gian giữa các snapshot (giây)
            depth: Độ sâu của orderbook
            duration: Thời gian thu thập (giây)
            is_futures: Sử dụng thị trường futures
            output_dir: Thư mục lưu dữ liệu
            
        Returns:
            True nếu thành công, False nếu có lỗi
        """
        self.logger.info(f"Thu thập dữ liệu orderbook cho {len(symbols)} cặp giao dịch từ {exchange_id}")
        
        # Xác định thư mục đầu ra
        if output_dir is None:
            # Tạo cấu trúc thư mục theo exchange/spot_futures/orderbook
            market_type = "futures" if is_futures else "spot"
            output_dir = self.data_dir / "orderbook" / exchange_id / market_type
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Lấy API key và secret từ biến môi trường
            api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
            api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
            
            # Tạo orderbook collector
            self.orderbook_collector = await create_orderbook_collector(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                sandbox=False,
                is_futures=is_futures,
                snapshot_interval=snapshot_interval
            )
            
            # Tạo callback để in cảnh báo thanh khoản
            def print_liquidity_alert(symbol, alert_data):
                alerts = alert_data.get('alerts', [])
                if alerts:
                    alert_msg = ", ".join([alert['message'] for alert in alerts])
                    self.logger.warning(f"Cảnh báo thanh khoản {symbol}: {alert_msg}")
            
            # Tạo màn hình theo dõi thanh khoản
            from data_collectors.market_data.orderbook_collector import MarketLiquidityMonitor
            monitor = MarketLiquidityMonitor(
                orderbook_collector=self.orderbook_collector,
                alert_threshold=5.0,
                window_size=10
            )
            
            # Đăng ký callback
            monitor.add_alert_callback(print_liquidity_alert)
            
            # Bắt đầu thu thập dữ liệu
            await self.orderbook_collector.start(symbols)
            
            self.logger.info(f"Đã bắt đầu thu thập dữ liệu orderbook cho {len(symbols)} cặp giao dịch")
            self.logger.info(f"Thu thập sẽ diễn ra trong {duration} giây...")
            
            # Đợi trong thời gian xác định
            await asyncio.sleep(duration)
            
            # Dừng thu thập
            await self.orderbook_collector.stop()
            
            self.logger.info("Đã hoàn thành thu thập dữ liệu orderbook")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập dữ liệu orderbook: {str(e)}")
            return False
        finally:
            # Đóng kết nối
            if self.orderbook_collector:
                try:
                    await self.orderbook_collector.stop()
                    await self.orderbook_collector.exchange_connector.close()
                except:
                    pass
                self.orderbook_collector = None
    
    async def collect_funding_rates(
        self,
        exchange_id: str,
        symbols: List[str],
        output_dir: Optional[Path] = None,
        save_format: str = 'parquet'
    ) -> Dict[str, Path]:
        """
        Thu thập dữ liệu tỷ lệ tài trợ (funding rates).
        
        Args:
            exchange_id: ID của sàn giao dịch
            symbols: Danh sách cặp giao dịch
            output_dir: Thư mục lưu dữ liệu
            save_format: Định dạng lưu trữ ('parquet', 'csv', 'json')
            
        Returns:
            Dict với key là symbol và value là đường dẫn file dữ liệu
        """
        self.logger.info(f"Thu thập tỷ lệ tài trợ cho {len(symbols)} cặp giao dịch từ {exchange_id}")
        
        # Xác định thư mục đầu ra
        if output_dir is None:
            output_dir = self.data_dir / "funding" / exchange_id
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            # Lấy API key và secret từ biến môi trường
            api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
            api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
            
            # Tạo collector
            self.historical_collector = await create_data_collector(
                exchange_id=exchange_id,
                api_key=api_key,
                api_secret=api_secret,
                testnet=False,
                is_futures=True,  # Funding rates chỉ có ở thị trường futures
                max_workers=4
            )
            
            # Thu thập tỷ lệ tài trợ cho tất cả các cặp giao dịch
            funding_data = await self.historical_collector.collect_all_funding_rates(
                symbols=symbols,
                save_format=save_format
            )
            
            # Tạo dict đường dẫn file
            result_files = {}
            
            for symbol in funding_data:
                if not funding_data[symbol].empty:
                    # Tạo tên file
                    filename = f"{symbol.replace('/', '_')}_funding".lower()
                    
                    if save_format == 'parquet':
                        file_path = output_dir / f"{filename}.parquet"
                    elif save_format == 'csv':
                        file_path = output_dir / f"{filename}.csv"
                    elif save_format == 'json':
                        file_path = output_dir / f"{filename}.json"
                    else:
                        file_path = output_dir / f"{filename}.parquet"
                    
                    result_files[symbol] = file_path
                    self.logger.info(f"Đã thu thập tỷ lệ tài trợ cho {symbol}")
                else:
                    self.logger.warning(f"Không có dữ liệu tỷ lệ tài trợ cho {symbol}")
            
            return result_files
            
        except Exception as e:
            self.logger.error(f"Lỗi khi thu thập tỷ lệ tài trợ: {str(e)}")
            return {}
        finally:
            # Đóng kết nối
            if self.historical_collector:
                await self.historical_collector.exchange_connector.close()
                self.historical_collector = None
    
    def get_available_exchanges(self) -> List[str]:
        """
        Lấy danh sách sàn giao dịch có sẵn.
        
        Returns:
            Danh sách các sàn giao dịch hỗ trợ
        """
        # Danh sách các sàn giao dịch được hỗ trợ
        exchanges = [
            "binance", "bybit", "kucoin", "okx", "huobi", "ftx", "kraken",
            "bitfinex", "bitmex", "coinbase", "bitstamp", "gate"
        ]
        
        return exchanges
    
    def get_available_timeframes(self) -> Dict[str, int]:
        """
        Lấy danh sách khung thời gian có sẵn.
        
        Returns:
            Dict với key là tên timeframe và value là số giây
        """
        from config.constants import TIMEFRAME_TO_SECONDS
        return TIMEFRAME_TO_SECONDS
    
    def get_data_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dữ liệu đã thu thập.
        
        Returns:
            Dict chứa thông tin về dữ liệu
        """
        info = {
            "data_dir": str(self.data_dir),
            "exchanges": {},
            "symbols": set(),
            "timeframes": set()
        }
        
        # Tìm tất cả các thư mục sàn giao dịch
        exchanges_dirs = [d for d in self.data_dir.glob("*/*") if d.is_dir()]
        
        for exchange_dir in exchanges_dirs:
            exchange_name = exchange_dir.name
            if exchange_name not in info["exchanges"]:
                info["exchanges"][exchange_name] = {}
            
            # Tìm các file dữ liệu
            data_files = []
            for ext in [".parquet", ".csv", ".json"]:
                data_files.extend(list(exchange_dir.glob(f"**/*{ext}")))
            
            # Phân tích tên file để lấy thông tin
            for file_path in data_files:
                file_name = file_path.stem
                parts = file_name.split("_")
                
                if len(parts) >= 2:
                    symbol = parts[0]
                    
                    # Thêm vào danh sách symbols
                    info["symbols"].add(symbol)
                    
                    # Nếu có timeframe
                    if len(parts) >= 3 and parts[1] in ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"]:
                        timeframe = parts[1]
                        info["timeframes"].add(timeframe)
                        
                        # Thêm vào thông tin sàn giao dịch
                        if "data" not in info["exchanges"][exchange_name]:
                            info["exchanges"][exchange_name]["data"] = {}
                        
                        if symbol not in info["exchanges"][exchange_name]["data"]:
                            info["exchanges"][exchange_name]["data"][symbol] = []
                        
                        info["exchanges"][exchange_name]["data"][symbol].append({
                            "timeframe": timeframe,
                            "file": str(file_path),
                            "size": file_path.stat().st_size
                        })
        
        # Chuyển set thành list
        info["symbols"] = list(info["symbols"])
        info["timeframes"] = list(info["timeframes"])
        
        return info

def setup_collect_parser(subparsers):
    """
    Thiết lập parser cho lệnh 'collect'.
    
    Args:
        subparsers: Đối tượng subparsers từ argparse
    """
    collect_parser = subparsers.add_parser(
        'collect',
        help='Thu thập dữ liệu thị trường',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các tham số chung cho cú pháp trực tiếp
    collect_parser.add_argument(
        '--exchange',
        type=str,
        help='Tên sàn giao dịch (binance, bybit, ...)'
    )
    
    collect_parser.add_argument(
        '--symbols',
        type=str,
        help='Danh sách cặp giao dịch, ngăn cách bởi dấu phẩy'
    )
    
    collect_parser.add_argument(
        '--timeframes',
        type=str,
        help='Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d...), phân cách bằng dấu phẩy'
    )
    
    collect_parser.add_argument(
        '--start-date',
        type=str,
        help='Ngày bắt đầu (định dạng YYYY-MM-DD)'
    )
    
    collect_parser.add_argument(
        '--end-date',
        type=str,
        help='Ngày kết thúc (định dạng YYYY-MM-DD)'
    )
    
    collect_parser.add_argument(
        '--futures',
        action='store_true',
        help='Thu thập dữ liệu futures thay vì spot'
    )
    
    collect_parser.add_argument(
        '--output-dir',
        type=str,
        help='Thư mục lưu dữ liệu'
    )
    
    collect_parser.add_argument(
        '--save-format',
        type=str,
        choices=['parquet', 'csv', 'json'],
        default='parquet',
        help='Định dạng lưu trữ'
    )
    
    collect_parser.add_argument(
        '--force-update',
        action='store_true',
        help='Cập nhật dữ liệu kể cả khi đã có'
    )
    
    collect_parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Số luồng tối đa'
    )
    
    # Tạo subparsers cho các lệnh thu thập
    # Sử dụng required=False để cho phép lệnh collect hoạt động mà không cần subcommand
    collect_subparsers = collect_parser.add_subparsers(
        title='collect_command',
        description='Các lệnh thu thập dữ liệu',
        dest='collect_command',
        required=False  # Chỉ hoạt động trên Python 3.7+
    )
    
    # Nếu bạn sử dụng Python 3.6 hoặc cũ hơn, thay dòng required=False ở trên bằng:
    # collect_parser.set_defaults(collect_command=None)
    
    # ===== THIẾT LẬP PARSER CHO HISTORICAL =====
    historical_parser = collect_subparsers.add_parser(
        'historical',
        help='Thu thập dữ liệu lịch sử',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các tham số cho lệnh thu thập dữ liệu lịch sử
    historical_parser.add_argument(
        '--exchange',
        type=str,
        required=True,
        help='Sàn giao dịch (binance, bybit, ...)'
    )
    
    historical_parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Danh sách cặp giao dịch (phân cách bằng dấu phẩy)'
    )
    
    historical_parser.add_argument(
        '--timeframe',
        type=str,
        default='1h',
        help='Khung thời gian (1m, 5m, 15m, 1h, 4h, 1d, ...)'
    )
    
    historical_parser.add_argument(
        '--start-date',
        type=str,
        help='Ngày bắt đầu (định dạng YYYY-MM-DD)'
    )
    
    historical_parser.add_argument(
        '--end-date',
        type=str,
        help='Ngày kết thúc (định dạng YYYY-MM-DD)'
    )
    
    historical_parser.add_argument(
        '--futures',
        action='store_true',
        help='Sử dụng thị trường futures'
    )
    
    historical_parser.add_argument(
        '--output-dir',
        type=str,
        help='Thư mục lưu dữ liệu'
    )
    
    historical_parser.add_argument(
        '--save-format',
        type=str,
        choices=['parquet', 'csv', 'json'],
        default='parquet',
        help='Định dạng lưu trữ'
    )
    
    historical_parser.add_argument(
        '--force-update',
        action='store_true',
        help='Cập nhật dữ liệu kể cả khi đã có'
    )
    
    historical_parser.add_argument(
        '--max-workers',
        type=int,
        default=4,
        help='Số luồng tối đa'
    )
    
    # Thiết lập hàm xử lý
    historical_parser.set_defaults(func=handle_historical_command)
    
    # ===== THIẾT LẬP PARSER CHO REALTIME =====
    realtime_parser = collect_subparsers.add_parser(
        'realtime',
        help='Thu thập dữ liệu thời gian thực',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các tham số cho lệnh thu thập dữ liệu thời gian thực
    realtime_parser.add_argument(
        '--exchange',
        type=str,
        required=True,
        help='Sàn giao dịch (binance, bybit, ...)'
    )
    
    realtime_parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Danh sách cặp giao dịch (phân cách bằng dấu phẩy)'
    )
    
    realtime_parser.add_argument(
        '--channels',
        type=str,
        default='ticker,kline_1m,trade',
        help='Danh sách kênh dữ liệu (phân cách bằng dấu phẩy)'
    )
    
    realtime_parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Thời gian thu thập (giây)'
    )
    
    realtime_parser.add_argument(
        '--futures',
        action='store_true',
        help='Sử dụng thị trường futures'
    )
    
    realtime_parser.add_argument(
        '--output-dir',
        type=str,
        help='Thư mục lưu dữ liệu'
    )
    
    # Thiết lập hàm xử lý
    realtime_parser.set_defaults(func=handle_realtime_command)
    
    # ===== THIẾT LẬP PARSER CHO ORDERBOOK =====
    orderbook_parser = collect_subparsers.add_parser(
        'orderbook',
        help='Thu thập dữ liệu orderbook',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các tham số cho lệnh thu thập dữ liệu orderbook
    orderbook_parser.add_argument(
        '--exchange',
        type=str,
        required=True,
        help='Sàn giao dịch (binance, bybit, ...)'
    )
    
    orderbook_parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Danh sách cặp giao dịch (phân cách bằng dấu phẩy)'
    )
    
    orderbook_parser.add_argument(
        '--interval',
        type=int,
        default=60,
        help='Thời gian giữa các snapshot (giây)'
    )
    
    orderbook_parser.add_argument(
        '--depth',
        type=int,
        default=20,
        help='Độ sâu của orderbook'
    )
    
    orderbook_parser.add_argument(
        '--duration',
        type=int,
        default=3600,
        help='Thời gian thu thập (giây)'
    )
    
    orderbook_parser.add_argument(
        '--futures',
        action='store_true',
        help='Sử dụng thị trường futures'
    )
    
    orderbook_parser.add_argument(
        '--output-dir',
        type=str,
        help='Thư mục lưu dữ liệu'
    )
    
    # Thiết lập hàm xử lý
    orderbook_parser.set_defaults(func=handle_orderbook_command)
    
    # ===== THIẾT LẬP PARSER CHO FUNDING =====
    funding_parser = collect_subparsers.add_parser(
        'funding',
        help='Thu thập tỷ lệ tài trợ',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thêm các tham số cho lệnh thu thập tỷ lệ tài trợ
    funding_parser.add_argument(
        '--exchange',
        type=str,
        required=True,
        help='Sàn giao dịch (binance, bybit, ...)'
    )
    
    funding_parser.add_argument(
        '--symbols',
        type=str,
        required=True,
        help='Danh sách cặp giao dịch (phân cách bằng dấu phẩy)'
    )
    
    funding_parser.add_argument(
        '--output-dir',
        type=str,
        help='Thư mục lưu dữ liệu'
    )
    
    funding_parser.add_argument(
        '--save-format',
        type=str,
        choices=['parquet', 'csv', 'json'],
        default='parquet',
        help='Định dạng lưu trữ'
    )
    
    # Thiết lập hàm xử lý
    funding_parser.set_defaults(func=handle_funding_command)
    
    # ===== THIẾT LẬP PARSER CHO INFO =====
    info_parser = collect_subparsers.add_parser(
        'info',
        help='Hiển thị thông tin về dữ liệu đã thu thập',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Thiết lập hàm xử lý
    info_parser.set_defaults(func=handle_info_command)
    
    # Thiết lập hàm xử lý mặc định cho lệnh collect
    collect_parser.set_defaults(func=handle_collect_command)

def handle_collect_command(args, system):
    """
    Xử lý lệnh 'collect' chung.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    # Kiểm tra xem có subcommand không
    if hasattr(args, 'collect_command') and args.collect_command:
        # Có subcommand, sử dụng handler tương ứng
        if hasattr(args, 'func') and args.func != handle_collect_command:
            # Gọi hàm xử lý tương ứng
            return args.func(args, system)
        return 0
    
    # Nếu không có subcommand, kiểm tra xem có đủ tham số cần thiết không
    if not hasattr(args, 'exchange') or not args.exchange:
        print("Lỗi: Thiếu tham số --exchange")
        print("Sử dụng: main.py collect --exchange <tên_sàn> --symbols <cặp_giao_dịch> [các tùy chọn]")
        return 1
    
    if not hasattr(args, 'symbols') or not args.symbols:
        print("Lỗi: Thiếu tham số --symbols")
        print("Sử dụng: main.py collect --exchange <tên_sàn> --symbols <cặp_giao_dịch> [các tùy chọn]")
        return 1
    
    # Xử lý các tham số
    exchange_id = args.exchange
    symbols = args.symbols.split(',')
    timeframes_str = args.timeframes if hasattr(args, 'timeframes') and args.timeframes else "1h"
    timeframes = timeframes_str.split(',')
    start_date = args.start_date if hasattr(args, 'start_date') else None
    end_date = args.end_date if hasattr(args, 'end_date') else None
    is_futures = args.futures if hasattr(args, 'futures') else False
    output_dir = Path(args.output_dir) if hasattr(args, 'output_dir') and args.output_dir else None
    save_format = args.save_format if hasattr(args, 'save_format') else 'parquet'
    force_update = args.force_update if hasattr(args, 'force_update') else False
    max_workers = args.max_workers if hasattr(args, 'max_workers') else 4
    
    # Tạo CollectCommands
    cmd = CollectCommands(system)
    
    try:
        # Chuẩn bị loop asyncio
        loop = asyncio.get_event_loop()
        
        # Tập hợp kết quả cho tất cả timeframes
        all_results = {}
        
        print(f"Thu thập dữ liệu lịch sử từ {exchange_id} cho {len(symbols)} cặp giao dịch...")
        
        for tf in timeframes:
            print(f"Đang thu thập dữ liệu với khung thời gian {tf}...")
            
            # Sử dụng phương thức collect_data của system nếu có
            if system and hasattr(system, 'collect_data'):
                result = loop.run_until_complete(system.collect_data(
                    exchange_id=exchange_id,
                    symbols=symbols,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                    futures=is_futures,
                    output_dir=output_dir
                ))
            else:
                # Nếu không, sử dụng phương thức collect_historical_data của CollectCommands
                result = loop.run_until_complete(cmd.collect_historical_data(
                    exchange_id=exchange_id,
                    symbols=symbols,
                    timeframe=tf,
                    start_date=start_date,
                    end_date=end_date,
                    is_futures=is_futures,
                    output_dir=output_dir,
                    save_format=save_format,
                    force_update=force_update,
                    max_workers=max_workers
                ))
            
            # Thêm vào kết quả tổng hợp
            for symbol, path in result.items():
                if symbol not in all_results:
                    all_results[symbol] = {}
                all_results[symbol][tf] = path
        
        # In kết quả
        if all_results:
            print(f"\nĐã thu thập dữ liệu cho {len(all_results)} cặp giao dịch:")
            for symbol, timeframe_paths in all_results.items():
                print(f"  {symbol}:")
                for tf, path in timeframe_paths.items():
                    print(f"    - {tf}: {path}")
            return 0
        else:
            print("Không có dữ liệu nào được thu thập.")
            return 1
            
    except Exception as e:
        print(f"Lỗi khi thu thập dữ liệu: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

def handle_historical_command(args, system):
    """
    Xử lý lệnh 'collect historical'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    # Tạo CollectCommands
    cmd = CollectCommands(system)
    
    # Phân tích tham số
    exchange_id = args.exchange
    symbols = args.symbols.split(',')
    timeframe = args.timeframe
    is_futures = args.futures
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Kiểm tra API keys
    api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
    api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
    
    if not api_key or not api_secret:
        print(f"Cảnh báo: Không tìm thấy API key/secret cho {exchange_id}.")
        print("Sử dụng lệnh sau để cấu hình API key:")
        print(f"  export {exchange_id.upper()}_API_KEY=<api_key>")
        print(f"  export {exchange_id.upper()}_API_SECRET=<api_secret>")
        
        response = input("Bạn có muốn tiếp tục mà không có API key? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    # Sử dụng phương thức collect_data của system nếu có sẵn
    if system and hasattr(system, 'collect_data'):
        try:
            # Chuẩn bị loop asyncio
            loop = asyncio.get_event_loop()
            
            # Gọi phương thức collect_data
            result = loop.run_until_complete(system.collect_data(
                exchange_id=exchange_id,
                symbols=symbols,
                timeframe=timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                futures=is_futures,
                output_dir=output_dir
            ))
            
            # In kết quả
            if result:
                print(f"\nĐã thu thập dữ liệu cho {len(result)} cặp giao dịch:")
                for symbol, path in result.items():
                    print(f"  {symbol}: {path}")
                return 0
            else:
                print("Không có dữ liệu nào được thu thập.")
                return 1
                
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return 1
    else:
        # Sử dụng phương thức của CollectCommands
        try:
            # Chuẩn bị loop asyncio
            loop = asyncio.get_event_loop()
            
            # Gọi phương thức collect_historical_data
            result = loop.run_until_complete(cmd.collect_historical_data(
                exchange_id=exchange_id,
                symbols=symbols,
                timeframe=timeframe,
                start_date=args.start_date,
                end_date=args.end_date,
                is_futures=is_futures,
                output_dir=output_dir,
                save_format=args.save_format,
                force_update=args.force_update,
                max_workers=args.max_workers
            ))
            
            # In kết quả
            if result:
                print(f"\nĐã thu thập dữ liệu cho {len(result)} cặp giao dịch:")
                for symbol, path in result.items():
                    print(f"  {symbol}: {path}")
                return 0
            else:
                print("Không có dữ liệu nào được thu thập.")
                return 1
                
        except Exception as e:
            print(f"Lỗi: {str(e)}")
            return 1

def handle_realtime_command(args, system):
    """
    Xử lý lệnh 'collect realtime'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    # Tạo CollectCommands
    cmd = CollectCommands(system)
    
    # Phân tích tham số
    exchange_id = args.exchange
    symbols = args.symbols.split(',')
    channels = args.channels.split(',')
    duration = args.duration
    is_futures = args.futures
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Kiểm tra API keys
    api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
    api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
    
    if not api_key or not api_secret:
        print(f"Cảnh báo: Không tìm thấy API key/secret cho {exchange_id}.")
        print("Sử dụng lệnh sau để cấu hình API key:")
        print(f"  export {exchange_id.upper()}_API_KEY=<api_key>")
        print(f"  export {exchange_id.upper()}_API_SECRET=<api_secret>")
        
        response = input("Bạn có muốn tiếp tục mà không có API key? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    try:
        # Chuẩn bị loop asyncio
        loop = asyncio.get_event_loop()
        
        print(f"Bắt đầu thu thập dữ liệu thời gian thực cho {len(symbols)} cặp giao dịch...")
        print(f"Thu thập sẽ diễn ra trong {duration} giây. Nhấn Ctrl+C để dừng.")
        
        try:
            # Gọi phương thức collect_realtime_data
            result = loop.run_until_complete(cmd.collect_realtime_data(
                exchange_id=exchange_id,
                symbols=symbols,
                channels=channels,
                duration=duration,
                is_futures=is_futures,
                output_dir=output_dir
            ))
            
            if result:
                print("\nĐã hoàn thành thu thập dữ liệu thời gian thực.")
                return 0
            else:
                print("Có lỗi khi thu thập dữ liệu thời gian thực.")
                return 1
        
        except KeyboardInterrupt:
            print("\nĐã hủy thu thập dữ liệu thời gian thực.")
            return 130
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return 1

def handle_orderbook_command(args, system):
    """
    Xử lý lệnh 'collect orderbook'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    # Tạo CollectCommands
    cmd = CollectCommands(system)
    
    # Phân tích tham số
    exchange_id = args.exchange
    symbols = args.symbols.split(',')
    interval = args.interval
    depth = args.depth
    duration = args.duration
    is_futures = args.futures
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    # Kiểm tra API keys
    api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
    api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
    
    if not api_key or not api_secret:
        print(f"Cảnh báo: Không tìm thấy API key/secret cho {exchange_id}.")
        print("Sử dụng lệnh sau để cấu hình API key:")
        print(f"  export {exchange_id.upper()}_API_KEY=<api_key>")
        print(f"  export {exchange_id.upper()}_API_SECRET=<api_secret>")
        
        response = input("Bạn có muốn tiếp tục mà không có API key? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    try:
        # Chuẩn bị loop asyncio
        loop = asyncio.get_event_loop()
        
        print(f"Bắt đầu thu thập dữ liệu orderbook cho {len(symbols)} cặp giao dịch...")
        print(f"Thu thập sẽ diễn ra trong {duration} giây. Nhấn Ctrl+C để dừng.")
        
        try:
            # Gọi phương thức collect_orderbook_data
            result = loop.run_until_complete(cmd.collect_orderbook_data(
                exchange_id=exchange_id,
                symbols=symbols,
                snapshot_interval=interval,
                depth=depth,
                duration=duration,
                is_futures=is_futures,
                output_dir=output_dir
            ))
            
            if result:
                print("\nĐã hoàn thành thu thập dữ liệu orderbook.")
                return 0
            else:
                print("Có lỗi khi thu thập dữ liệu orderbook.")
                return 1
        
        except KeyboardInterrupt:
            print("\nĐã hủy thu thập dữ liệu orderbook.")
            return 130
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return 1

def handle_funding_command(args, system):
    """
    Xử lý lệnh 'collect funding'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    # Tạo CollectCommands
    cmd = CollectCommands(system)
    
    # Phân tích tham số
    exchange_id = args.exchange
    symbols = args.symbols.split(',')
    output_dir = Path(args.output_dir) if args.output_dir else None
    save_format = args.save_format
    
    # Kiểm tra API keys
    api_key = get_env(f"{exchange_id.upper()}_API_KEY", "")
    api_secret = get_env(f"{exchange_id.upper()}_API_SECRET", "")
    
    if not api_key or not api_secret:
        print(f"Cảnh báo: Không tìm thấy API key/secret cho {exchange_id}.")
        print("Sử dụng lệnh sau để cấu hình API key:")
        print(f"  export {exchange_id.upper()}_API_KEY=<api_key>")
        print(f"  export {exchange_id.upper()}_API_SECRET=<api_secret>")
        
        response = input("Bạn có muốn tiếp tục mà không có API key? (y/n): ")
        if response.lower() != 'y':
            return 1
    
    try:
        # Chuẩn bị loop asyncio
        loop = asyncio.get_event_loop()
        
        # Gọi phương thức collect_funding_rates
        result = loop.run_until_complete(cmd.collect_funding_rates(
            exchange_id=exchange_id,
            symbols=symbols,
            output_dir=output_dir,
            save_format=save_format
        ))
        
        # In kết quả
        if result:
            print(f"\nĐã thu thập tỷ lệ tài trợ cho {len(result)} cặp giao dịch:")
            for symbol, path in result.items():
                print(f"  {symbol}: {path}")
            return 0
        else:
            print("Không có dữ liệu nào được thu thập.")
            return 1
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return 1

def handle_info_command(args, system):
    """
    Xử lý lệnh 'collect info'.
    
    Args:
        args: Các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã kết quả (0 = thành công)
    """
    # Tạo CollectCommands
    cmd = CollectCommands(system)
    
    try:
        # Lấy thông tin về dữ liệu
        info = cmd.get_data_info()
        
        # Hiển thị thông tin
        print("\n=== THÔNG TIN DỮ LIỆU ===")
        print(f"Thư mục dữ liệu: {info['data_dir']}")
        
        if info['exchanges']:
            print("\nSàn giao dịch:")
            for exchange, exchange_info in info['exchanges'].items():
                print(f"  {exchange}:")
                
                if 'data' in exchange_info:
                    print(f"    Số cặp giao dịch: {len(exchange_info['data'])}")
                    
                    # Hiển thị một số cặp đầu tiên
                    for i, (symbol, symbol_data) in enumerate(exchange_info['data'].items()):
                        if i >= 5:  # Chỉ hiển thị 5 cặp đầu tiên
                            print(f"    ... và {len(exchange_info['data']) - 5} cặp khác")
                            break
                        
                        print(f"    - {symbol}: {len(symbol_data)} file")
                else:
                    print("    Không có dữ liệu")
        else:
            print("\nKhông có dữ liệu nào được thu thập.")
        
        if info['timeframes']:
            print("\nKhung thời gian có sẵn:")
            print(f"  {', '.join(info['timeframes'])}")
        
        return 0
            
    except Exception as e:
        print(f"Lỗi: {str(e)}")
        return 1

if __name__ == "__main__":
    # Khi chạy trực tiếp file này
    parser = argparse.ArgumentParser(description='Thu thập dữ liệu thị trường')
    setup_collect_parser(parser.add_subparsers())
    args = parser.parse_args()
    
    # Kiểm tra xem có tham số nào được chỉ định không
    if len(sys.argv) <= 1:
        parser.print_help()
        sys.exit(1)
    
    # Gọi hàm tương ứng
    if hasattr(args, 'func'):
        sys.exit(args.func(args, None))
    else:
        sys.exit(1)