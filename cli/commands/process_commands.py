"""
CLI commands cho xử lý dữ liệu.
File này cung cấp các lệnh để xử lý dữ liệu thị trường, bao gồm
làm sạch dữ liệu, tạo đặc trưng, và chạy pipeline xử lý đầy đủ.
"""

import os
import sys
import click
import json
import logging
import asyncio
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.system_config import get_system_config
from config.constants import Exchange, Timeframe, EXCHANGE_TIMEFRAMES
from config.utils.validators import is_valid_timeframe, is_valid_trading_pair, validate_config

from data_processors.data_pipeline import DataPipeline
from data_processors.feature_engineering.feature_generator import FeatureGenerator

# Thiết lập logger
logger = get_logger("process_commands")

# Lấy cấu hình hệ thống
system_config = get_system_config()

# Thư mục gốc của dự án
BASE_DIR = Path(__file__).parent.parent.parent

# Mặc định cho thư mục dữ liệu
DEFAULT_DATA_DIR = BASE_DIR / "data"
DEFAULT_OUTPUT_DIR = BASE_DIR / "data/processed"

# Định nghĩa nhóm lệnh process
@click.group(name="process")
def process_commands():
    """Các lệnh cho xử lý dữ liệu thị trường."""
    pass

@process_commands.command(name="clean")
@click.option("--data-type", "-t", type=click.Choice(['ohlcv', 'trades', 'orderbook', 'all']), default='ohlcv', help="Loại dữ liệu cần làm sạch")
@click.option("--input-dir", "-i", type=click.Path(exists=True), help="Thư mục chứa dữ liệu đầu vào")
@click.option("--symbols", "-s", multiple=True, help="Danh sách cặp giao dịch cần xử lý")
@click.option("--timeframes", "-tf", multiple=True, default=['1h'], help="Khung thời gian cần xử lý")
@click.option("--output-dir", "-o", type=click.Path(), help="Thư mục lưu dữ liệu đã làm sạch")
@click.option("--preserve-timestamp/--no-preserve-timestamp", default=True, help="Giữ nguyên timestamp trong quá trình xử lý")
@click.option("--verbose", "-v", count=True, help="Mức độ chi tiết của log (0-2)")
def clean_data(data_type, input_dir, symbols, timeframes, output_dir, preserve_timestamp, verbose):
    """
    Làm sạch dữ liệu thị trường.
    
    Lệnh này làm sạch dữ liệu thô, bao gồm loại bỏ nhiễu, điền giá trị thiếu, và chuẩn hóa định dạng.
    """
    try:
        # Thiết lập mức độ chi tiết
        log_level = _get_log_level(verbose)
        
        # Tạo data pipeline
        pipeline = DataPipeline(logger=logger)
        
        # Chuẩn bị thư mục đầu vào/ra
        input_dir_path, output_dir_path = _prepare_directories(input_dir, output_dir, default_input_subdir="collected")
        
        # Tìm các file dữ liệu
        data_paths = _find_data_files(input_dir_path, symbols, timeframes, logger)
        
        if not data_paths:
            logger.error(f"Không tìm thấy file dữ liệu phù hợp trong {input_dir_path}")
            return 1
        
        # Tải dữ liệu
        loaded_data = {}
        for symbol, path in data_paths.items():
            logger.info(f"Đang tải dữ liệu từ {path} cho {symbol}...")
            symbol_data = pipeline.load_data(
                file_paths=path,
                file_format=path.suffix[1:] if path.suffix else 'csv'
            )
            
            if symbol_data:
                # Nếu symbol_data có nhiều symbols, chỉ lấy symbol đúng
                if symbol in symbol_data:
                    loaded_data[symbol] = symbol_data[symbol]
                else:
                    # Lấy symbol đầu tiên nếu không tìm thấy
                    first_symbol = next(iter(symbol_data.keys()))
                    loaded_data[symbol] = symbol_data[first_symbol]
                    logger.warning(f"Không tìm thấy dữ liệu cho {symbol}, sử dụng {first_symbol} thay thế")
            else:
                logger.warning(f"Không thể tải dữ liệu cho {symbol}")
        
        if not loaded_data:
            logger.error("Không có dữ liệu nào được tải thành công")
            return 1
        
        # Làm sạch dữ liệu theo loại
        cleaned_data = {}
        if data_type == 'ohlcv' or data_type == 'all':
            cleaned_data = pipeline.clean_data(
                loaded_data,
                clean_ohlcv=True,
                clean_orderbook=(data_type == 'all'),
                clean_trades=(data_type == 'all'),
                preserve_timestamp=preserve_timestamp
            )
        elif data_type == 'trades':
            cleaned_data = pipeline.clean_data(
                loaded_data,
                clean_ohlcv=False,
                clean_trades=True,
                preserve_timestamp=preserve_timestamp
            )
        elif data_type == 'orderbook':
            cleaned_data = pipeline.clean_data(
                loaded_data,
                clean_ohlcv=False,
                clean_orderbook=True,
                preserve_timestamp=preserve_timestamp
            )
        
        if not cleaned_data:
            logger.error("Không có dữ liệu nào được làm sạch thành công")
            return 1
        
        # Lưu dữ liệu đã làm sạch
        saved_paths = pipeline.save_data(
            cleaned_data,
            output_dir=output_dir_path,
            file_format='parquet',
            include_metadata=True,
            preserve_timestamp=preserve_timestamp
        )
        
        # Hiển thị kết quả
        if saved_paths:
            click.echo("\n" + "="*50)
            click.echo(f"Kết quả làm sạch dữ liệu {data_type}:")
            for symbol, path in saved_paths.items():
                click.echo(f"  - {symbol}: {path}")
            click.echo("="*50 + "\n")
            
            return 0
        else:
            logger.error("Không có dữ liệu nào được lưu")
            return 1
            
    except Exception as e:
        logger.error(f"Lỗi khi làm sạch dữ liệu: {str(e)}", exc_info=True)
        return 1

@process_commands.command(name="features")
@click.option("--data-type", "-t", type=click.Choice(['ohlcv', 'all']), default='ohlcv', help="Loại dữ liệu cần tạo đặc trưng")
@click.option("--input-dir", "-i", type=click.Path(exists=True), help="Thư mục chứa dữ liệu đầu vào")
@click.option("--symbols", "-s", multiple=True, help="Danh sách cặp giao dịch cần xử lý")
@click.option("--indicators", multiple=True, help="Danh sách chỉ báo kỹ thuật cần tạo")
@click.option("--all-indicators/--selected-indicators", default=False, help="Tạo tất cả các chỉ báo kỹ thuật có sẵn")
@click.option("--output-dir", "-o", type=click.Path(), help="Thư mục lưu dữ liệu đã tạo đặc trưng")
@click.option("--remove-redundant/--keep-all", default=True, help="Loại bỏ các chỉ báo dư thừa")
@click.option("--generate-labels/--no-labels", default=True, help="Tạo nhãn cho huấn luyện có giám sát")
@click.option("--preserve-timestamp/--no-preserve-timestamp", default=True, help="Giữ nguyên timestamp trong quá trình xử lý")
@click.option("--verbose", "-v", count=True, help="Mức độ chi tiết của log (0-2)")
def create_features(data_type, input_dir, symbols, indicators, all_indicators, output_dir, remove_redundant, generate_labels, preserve_timestamp, verbose):
    """
    Tạo đặc trưng từ dữ liệu thị trường.
    
    Lệnh này tạo các đặc trưng kỹ thuật như các chỉ báo kỹ thuật (SMA, RSI, MACD, v.v.)
    và các đặc trưng khác từ dữ liệu thị trường đã làm sạch.
    """
    try:
        # Thiết lập mức độ chi tiết
        log_level = _get_log_level(verbose)
        
        # Tạo data pipeline
        pipeline = DataPipeline(logger=logger)
        
        # Chuẩn bị thư mục đầu vào/ra
        default_input = "processed" if not input_dir else None
        input_dir_path, output_dir_path = _prepare_directories(input_dir, output_dir, default_input_subdir=default_input)
        
        # Nếu thư mục processed không tồn tại, thử sử dụng thư mục collected
        if not input_dir_path.exists():
            input_dir_path, _ = _prepare_directories(None, None, default_input_subdir="collected")
            logger.info(f"Thư mục processed không tồn tại, sử dụng thư mục {input_dir_path}")
        
        # Tìm các file dữ liệu
        data_paths = _find_data_files(input_dir_path, symbols, [''], logger)  # Không filter theo timeframe
        
        if not data_paths:
            logger.error(f"Không tìm thấy file dữ liệu phù hợp trong {input_dir_path}")
            return 1
        
        # Tải dữ liệu
        loaded_data = {}
        for symbol, path in data_paths.items():
            logger.info(f"Đang tải dữ liệu từ {path} cho {symbol}...")
            symbol_data = pipeline.load_data(
                file_paths=path,
                file_format=path.suffix[1:] if path.suffix else 'csv'
            )
            
            if symbol_data:
                if symbol in symbol_data:
                    loaded_data[symbol] = symbol_data[symbol]
                else:
                    # Lấy symbol đầu tiên nếu không tìm thấy
                    first_symbol = next(iter(symbol_data.keys()))
                    loaded_data[symbol] = symbol_data[first_symbol]
                    logger.warning(f"Không tìm thấy dữ liệu cho {symbol}, sử dụng {first_symbol} thay thế")
            else:
                logger.warning(f"Không thể tải dữ liệu cho {symbol}")
        
        if not loaded_data:
            logger.error("Không có dữ liệu nào được tải thành công")
            return 1
        
        # Tạo cấu hình cho việc tạo đặc trưng
        feature_configs = {}
        for symbol in loaded_data.keys():
            feature_configs[symbol] = {
                "feature_names": list(indicators) if indicators else None,
                "generate_labels": generate_labels,
                "label_window": 10,
                "label_threshold": 0.01
            }
        
        # Tạo đặc trưng
        featured_data = pipeline.generate_features(
            loaded_data,
            feature_configs=feature_configs,
            all_indicators=all_indicators,
            preserve_timestamp=preserve_timestamp
        )
        
        if not featured_data:
            logger.error("Không có đặc trưng nào được tạo thành công")
            return 1
        
        # Loại bỏ các chỉ báo dư thừa nếu cần
        if remove_redundant:
            featured_data = pipeline.remove_redundant_indicators(
                featured_data,
                correlation_threshold=0.95,
                redundant_groups=[
                    ['macd_line', 'macd_signal', 'macd_histogram'],
                    ['atr_14', 'atr_pct_14', 'atr_norm_14', 'atr_norm_14_std'],
                    ['bb_middle_20', 'sma_20', 'bb_upper_20', 'bb_lower_20', 'bb_percent_b_20'],
                    ['plus_di_14', 'minus_di_14', 'adx_14'],
                    ['volume', 'volume_log'],
                    ['rsi_14', 'rsi_14_norm']
                ]
            )
        
        # Tạo nhãn nếu cần
        if generate_labels:
            featured_data = pipeline.create_target_features(
                featured_data,
                price_column="close",
                target_types=["direction", "return", "volatility"],
                horizons=[1, 3, 5, 10],
                threshold=0.001
            )
        
        # Lưu dữ liệu đã tạo đặc trưng
        saved_paths = pipeline.save_data(
            featured_data,
            output_dir=output_dir_path,
            file_format='parquet',
            include_metadata=True,
            preserve_timestamp=preserve_timestamp
        )
        
        # Hiển thị kết quả
        if saved_paths:
            total_features = {symbol: len(df.columns) for symbol, df in featured_data.items()}
            
            click.echo("\n" + "="*50)
            click.echo(f"Kết quả tạo đặc trưng cho {len(saved_paths)} cặp tiền:")
            for symbol, path in saved_paths.items():
                click.echo(f"  - {symbol}: {total_features[symbol]} đặc trưng, lưu tại {path}")
            click.echo("="*50 + "\n")
            
            return 0
        else:
            logger.error("Không có dữ liệu nào được lưu")
            return 1
            
    except Exception as e:
        logger.error(f"Lỗi khi tạo đặc trưng: {str(e)}", exc_info=True)
        return 1

@process_commands.command(name="pipeline")
@click.option("--input-dir", "-i", type=click.Path(exists=True), help="Thư mục chứa dữ liệu đầu vào")
@click.option("--symbols", "-s", multiple=True, help="Danh sách cặp giao dịch cần xử lý")
@click.option("--timeframes", "-tf", multiple=True, default=['1h'], help="Khung thời gian cần xử lý")
@click.option("--start-date", type=str, help="Ngày bắt đầu (YYYY-MM-DD)")
@click.option("--end-date", type=str, help="Ngày kết thúc (YYYY-MM-DD)")
@click.option("--output-dir", "-o", type=click.Path(), help="Thư mục lưu dữ liệu đã xử lý")
@click.option("--pipeline-name", type=str, help="Tên của pipeline xử lý (nếu sử dụng pipeline đã đăng ký)")
@click.option("--no-clean/--clean", default=False, help="Bỏ qua bước làm sạch dữ liệu")
@click.option("--no-features/--features", default=False, help="Bỏ qua bước tạo đặc trưng")
@click.option("--all-indicators/--selected-indicators", default=True, help="Sử dụng tất cả các chỉ báo kỹ thuật có sẵn")
@click.option("--preserve-timestamp/--no-preserve-timestamp", default=True, help="Giữ nguyên timestamp trong quá trình xử lý")
@click.option("--verbose", "-v", count=True, help="Mức độ chi tiết của log (0-2)")
def run_pipeline(input_dir, symbols, timeframes, start_date, end_date, output_dir, pipeline_name, no_clean, no_features, all_indicators, preserve_timestamp, verbose):
    """
    Chạy toàn bộ pipeline xử lý dữ liệu.
    
    Lệnh này thực hiện toàn bộ quy trình xử lý dữ liệu từ làm sạch đến tạo đặc trưng,
    có thể sử dụng pipeline đã đăng ký hoặc cấu hình mới.
    """
    try:
        # Thiết lập mức độ chi tiết
        log_level = _get_log_level(verbose)
        
        # Tạo data pipeline
        pipeline = DataPipeline(logger=logger)
        
        # Chuẩn bị thư mục đầu vào/ra
        input_dir_path, output_dir_path = _prepare_directories(input_dir, output_dir, default_input_subdir="collected")
        
        # Chuyển đổi start_date và end_date thành datetime
        start_datetime = None
        end_datetime = None
        
        if start_date:
            try:
                start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
                logger.info(f"Dữ liệu từ ngày: {start_date}")
            except ValueError:
                logger.error(f"Định dạng ngày không hợp lệ: {start_date}, cần định dạng YYYY-MM-DD")
                return 1
                
        if end_date:
            try:
                end_datetime = datetime.strptime(end_date, "%Y-%m-%d")
                logger.info(f"Dữ liệu đến ngày: {end_date}")
            except ValueError:
                logger.error(f"Định dạng ngày không hợp lệ: {end_date}, cần định dạng YYYY-MM-DD")
                return 1
        
        # Tìm các file dữ liệu hoặc sử dụng pipeline.run_pipeline nếu pipeline_name được cung cấp
        if pipeline_name:
            # Nếu có pipeline_name, sử dụng run_pipeline trực tiếp
            logger.info(f"Sử dụng pipeline đã đăng ký: {pipeline_name}")
            
            # Kiểm tra xem có symbols
            if not symbols:
                symbols_list = None
            else:
                symbols_list = list(symbols)
            
            # Chuyển đổi timeframes thành list
            timeframes_list = list(timeframes)
            
            # Chạy pipeline
            result_data = asyncio.run(pipeline.run_pipeline(
                pipeline_name=pipeline_name,
                input_files=None,
                exchange_id=None,
                symbols=symbols_list,
                timeframe=timeframes_list[0] if timeframes_list else "1h",
                start_time=start_datetime,
                end_time=end_datetime,
                output_dir=output_dir_path,
                save_results=True,
                preserve_timestamp=preserve_timestamp
            ))
            
            if not result_data:
                logger.error("Không có kết quả từ pipeline")
                return 1
            
            # Hiển thị kết quả
            click.echo("\n" + "="*50)
            click.echo(f"Kết quả xử lý pipeline {pipeline_name}:")
            for symbol in result_data.keys():
                click.echo(f"  - {symbol}: {len(result_data[symbol])} dòng, {len(result_data[symbol].columns)} cột")
            click.echo("="*50 + "\n")
            
            return 0
        
        # Nếu không có pipeline_name, thực hiện xử lý theo từng bước
        data_paths = _find_data_files(input_dir_path, symbols, timeframes, logger)
        
        if not data_paths:
            logger.error(f"Không tìm thấy file dữ liệu phù hợp trong {input_dir_path}")
            return 1
        
        # Tải dữ liệu
        loaded_data = {}
        for symbol, path in data_paths.items():
            logger.info(f"Đang tải dữ liệu từ {path} cho {symbol}...")
            symbol_data = pipeline.load_data(
                file_paths=path,
                file_format=path.suffix[1:] if path.suffix else 'csv'
            )
            
            if symbol_data:
                if symbol in symbol_data:
                    loaded_data[symbol] = symbol_data[symbol]
                else:
                    # Lấy symbol đầu tiên nếu không tìm thấy
                    first_symbol = next(iter(symbol_data.keys()))
                    loaded_data[symbol] = symbol_data[first_symbol]
                    logger.warning(f"Không tìm thấy dữ liệu cho {symbol}, sử dụng {first_symbol} thay thế")
            else:
                logger.warning(f"Không thể tải dữ liệu cho {symbol}")
        
        if not loaded_data:
            logger.error("Không có dữ liệu nào được tải thành công")
            return 1
        
        # Xử lý dữ liệu theo từng bước
        processed_data = loaded_data
        
        # Làm sạch dữ liệu nếu cần
        if not no_clean:
            processed_data = pipeline.clean_data(
                processed_data,
                clean_ohlcv=True,
                clean_orderbook=False,
                clean_trades=False,
                preserve_timestamp=preserve_timestamp
            )
            
            if not processed_data:
                logger.error("Không có dữ liệu nào được làm sạch thành công")
                return 1
            
            logger.info(f"Đã làm sạch dữ liệu cho {len(processed_data)} cặp tiền")
        
        # Tạo đặc trưng nếu cần
        if not no_features:
            # Đăng ký các chỉ báo cần thiết cho Fear & Greed Index
            required_indicators = ["trend_bbands", "momentum_roc", "volatility_atr", "volume_obv"]
            feature_configs = {}
            for symbol in processed_data.keys():
                feature_configs[symbol] = {
                    "feature_names": required_indicators,
                    "params": {
                        "trend_bbands": {"timeperiod": 20},
                        "momentum_roc": {"timeperiods": [1, 5, 10, 20]}
                    }
                }
            # Tính toán các chỉ báo bắt buộc
            processed_data = pipeline.generate_features(
                processed_data,
                all_indicators=all_indicators,
                preserve_timestamp=preserve_timestamp
            )
            
            if not processed_data:
                logger.error("Không có đặc trưng nào được tạo thành công")
                return 1
            
            logger.info(f"Đã tạo đặc trưng cho {len(processed_data)} cặp tiền")
            
            # Tạo nhãn
            processed_data = pipeline.create_target_features(
                processed_data,
                price_column="close",
                target_types=["direction", "return", "volatility"],
                horizons=[1, 3, 5, 10],
                threshold=0.001
            )
            
            logger.info(f"Đã tạo nhãn mục tiêu cho {len(processed_data)} cặp tiền")
        
        # Kết hợp dữ liệu tâm lý nếu được yêu cầu
        if hasattr(args, 'include_sentiment') and args.include_sentiment:
            try:
                # Tìm file tâm lý
                sentiment_dir = DEFAULT_DATA_DIR / "sentiment" / "binance"
                sentiment_files = list(sentiment_dir.glob("*_sentiment_*.csv"))
                
                if sentiment_files:
                    # Lấy file mới nhất
                    newest_sentiment_file = max(sentiment_files, key=lambda x: x.stat().st_mtime)
                    logger.info(f"Tìm thấy file tâm lý: {newest_sentiment_file}")
                    
                    # Tải dữ liệu tâm lý
                    sentiment_data = pd.read_csv(newest_sentiment_file)
                    
                    # Đảm bảo có cột timestamp
                    if 'timestamp' in sentiment_data.columns:
                        sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
                        
                        # Kết hợp dữ liệu tâm lý với dữ liệu thị trường
                        processed_data = pipeline.merge_sentiment_data(
                            processed_data,
                            sentiment_data=sentiment_data,
                            method='last_value',
                            window='1D'
                        )
                        
                        logger.info(f"Đã kết hợp dữ liệu tâm lý từ {newest_sentiment_file}")
                    else:
                        logger.warning("File tâm lý không có cột timestamp, không thể kết hợp")
                else:
                    logger.warning(f"Không tìm thấy file tâm lý trong {sentiment_dir}")
            except Exception as e:
                logger.error(f"Lỗi khi kết hợp dữ liệu tâm lý: {str(e)}")

        # Lưu kết quả
        saved_paths = pipeline.save_data(
            processed_data,
            output_dir=output_dir_path,
            file_format='parquet',
            include_metadata=True,
            preserve_timestamp=preserve_timestamp
        )
        
        # Hiển thị kết quả
        if saved_paths:
            click.echo("\n" + "="*50)
            click.echo(f"Kết quả xử lý pipeline cho {len(saved_paths)} cặp tiền:")
            for symbol, path in saved_paths.items():
                click.echo(f"  - {symbol}: {len(processed_data[symbol])} dòng, {len(processed_data[symbol].columns)} cột")
                click.echo(f"    Lưu tại: {path}")
            click.echo("="*50 + "\n")
            
            return 0
        else:
            logger.error("Không có dữ liệu nào được lưu")
            return 1
            
    except Exception as e:
        logger.error(f"Lỗi khi chạy pipeline xử lý dữ liệu: {str(e)}", exc_info=True)
        return 1

def _get_log_level(verbose_count: int) -> int:
    """
    Xác định mức độ log dựa trên số lượng -v.
    
    Args:
        verbose_count: Số lượng lần sử dụng flag -v
    
    Returns:
        Mức độ log (logging.DEBUG, logging.INFO, logging.WARNING)
    """
    if verbose_count >= 2:
        return logging.DEBUG
    elif verbose_count == 1:
        return logging.INFO
    else:
        return logging.WARNING

def _prepare_directories(input_dir: Optional[str], 
                         output_dir: Optional[str], 
                         default_input_subdir: str = "collected") -> Tuple[Path, Path]:
    """
    Chuẩn bị và kiểm tra các thư mục đầu vào/đầu ra.
    
    Args:
        input_dir: Thư mục đầu vào
        output_dir: Thư mục đầu ra
        default_input_subdir: Thư mục con mặc định
        
    Returns:
        Tuple (input_dir_path, output_dir_path)
    """
    # Chuyển đổi thành đường dẫn nếu có
    if input_dir:
        input_dir_path = Path(input_dir)
    else:
        input_dir_path = DEFAULT_DATA_DIR / default_input_subdir
    
    # Chuyển đổi output_dir nếu có
    if output_dir:
        output_dir_path = Path(output_dir)
    else:
        # Tạo thư mục đầu ra dựa trên loại xử lý
        if default_input_subdir == "collected":
            output_dir_path = DEFAULT_DATA_DIR / "processed"
        else:
            output_dir_path = DEFAULT_DATA_DIR / "features"
    
    # Đảm bảo thư mục đầu ra tồn tại
    output_dir_path.mkdir(parents=True, exist_ok=True)
    
    return input_dir_path, output_dir_path

def _find_data_files(input_dir: Path, 
                    symbols: Optional[Tuple[str, ...]], 
                    timeframes: Tuple[str, ...], 
                    logger: logging.Logger) -> Dict[str, Path]:
    """
    Tìm các file dữ liệu phù hợp với symbols và timeframes.
    """
    data_paths = {}
    
    # Duyệt qua các thư mục con để tìm file
    if input_dir.exists():
        # Thêm log để debug
        logger.info(f"Đang tìm kiếm dữ liệu trong thư mục: {input_dir}")
        
        # Liệt kê tất cả file parquet và csv trong thư mục và các thư mục con
        all_parquet_files = list(input_dir.glob("**/*.parquet"))
        all_csv_files = list(input_dir.glob("**/*.csv"))
        all_files = all_parquet_files + all_csv_files
        
        logger.info(f"Tìm thấy tổng cộng {len(all_files)} file dữ liệu")
        
        if symbols:
            for symbol in symbols:
                symbol_safe = symbol.replace('/', '_').lower()
                found = False
                
                # Tìm file phù hợp với symbol
                for file_path in all_files:
                    file_name = file_path.stem.lower()
                    if symbol_safe in file_name:
                        # Kiểm tra timeframe nếu có yêu cầu
                        if timeframes and any(tf.lower() in file_path.parent.name.lower() for tf in timeframes):
                            data_paths[symbol] = file_path
                            logger.info(f"Tìm thấy file cho {symbol}: {file_path}")
                            found = True
                            break
                        elif not timeframes:
                            data_paths[symbol] = file_path
                            logger.info(f"Tìm thấy file cho {symbol}: {file_path}")
                            found = True
                            break
                
                if not found:
                    logger.warning(f"Không tìm thấy file dữ liệu cho {symbol}")
        else:
            # Nếu không chỉ định symbols, lấy tất cả các file
            for file_path in all_files:
                file_name = file_path.stem.lower()
                parts = file_name.split('_')
                
                # Tìm symbol từ tên file (giả sử 2 phần đầu là symbol)
                if len(parts) >= 2:
                    symbol = f"{parts[0].upper()}/{parts[1].upper()}"
                    if symbol not in data_paths:
                        data_paths[symbol] = file_path
                        logger.info(f"Tìm thấy dữ liệu cho {symbol}: {file_path}")
    
    return data_paths

def setup_process_parser(subparsers):
    """
    Thiết lập parser cho các lệnh xử lý dữ liệu.
    
    Args:
        subparsers: Đối tượng subparsers từ argparse
    """
    # Tạo parser cho nhóm lệnh process
    process_parser = subparsers.add_parser('process', help='Các lệnh xử lý dữ liệu thị trường')
    process_subparsers = process_parser.add_subparsers(dest='process_command', help='Lệnh xử lý cụ thể')
    
    # Parser cho lệnh clean
    clean_parser = process_subparsers.add_parser('clean', help='Làm sạch dữ liệu thị trường')
    clean_parser.add_argument("--data-type", "-t", type=str, choices=['ohlcv', 'trades', 'orderbook', 'all'], default='ohlcv', help="Loại dữ liệu cần làm sạch")
    clean_parser.add_argument("--input-dir", "-i", type=str, help="Thư mục chứa dữ liệu đầu vào")
    clean_parser.add_argument("--symbols", "-s", nargs='+', help="Danh sách cặp giao dịch cần xử lý")
    clean_parser.add_argument("--timeframes", "-tf", nargs='+', default=['1h'], help="Khung thời gian cần xử lý")
    clean_parser.add_argument("--output-dir", "-o", type=str, help="Thư mục lưu dữ liệu đã làm sạch")
    clean_parser.add_argument("--preserve-timestamp/--no-preserve-timestamp", dest="preserve_timestamp", action="store_true", default=True, help="Giữ nguyên timestamp trong quá trình xử lý")
    clean_parser.add_argument("--verbose", "-v", action="count", default=0, help="Mức độ chi tiết của log (0-2)")
    
    # Parser cho lệnh features
    features_parser = process_subparsers.add_parser('features', help='Tạo đặc trưng từ dữ liệu thị trường')
    features_parser.add_argument("--data-type", "-t", type=str, choices=['ohlcv', 'all'], default='ohlcv', help="Loại dữ liệu cần tạo đặc trưng")
    features_parser.add_argument("--input-dir", "-i", type=str, help="Thư mục chứa dữ liệu đầu vào")
    features_parser.add_argument("--symbols", "-s", nargs='+', help="Danh sách cặp giao dịch cần xử lý")
    features_parser.add_argument("--indicators", nargs='+', help="Danh sách chỉ báo kỹ thuật cần tạo")
    features_parser.add_argument("--all-indicators/--selected-indicators", dest="all_indicators", action="store_true", default=False, help="Tạo tất cả các chỉ báo kỹ thuật có sẵn")
    features_parser.add_argument("--output-dir", "-o", type=str, help="Thư mục lưu dữ liệu đã tạo đặc trưng")
    features_parser.add_argument("--remove-redundant/--keep-all", dest="remove_redundant", action="store_true", default=True, help="Loại bỏ các chỉ báo dư thừa")
    features_parser.add_argument("--generate-labels/--no-labels", dest="generate_labels", action="store_true", default=True, help="Tạo nhãn cho huấn luyện có giám sát")
    features_parser.add_argument("--preserve-timestamp/--no-preserve-timestamp", dest="preserve_timestamp", action="store_true", default=True, help="Giữ nguyên timestamp trong quá trình xử lý")
    features_parser.add_argument("--verbose", "-v", action="count", default=0, help="Mức độ chi tiết của log (0-2)")
    
    # Parser cho lệnh pipeline
    pipeline_parser = process_subparsers.add_parser('pipeline', help='Chạy toàn bộ pipeline xử lý dữ liệu')
    pipeline_parser.add_argument("--input-dir", "-i", type=str, help="Thư mục chứa dữ liệu đầu vào")
    pipeline_parser.add_argument("--symbols", "-s", nargs='+', help="Danh sách cặp giao dịch cần xử lý")
    pipeline_parser.add_argument("--timeframes", "-tf", nargs='+', default=['1h'], help="Khung thời gian cần xử lý")
    pipeline_parser.add_argument("--start-date", type=str, help="Ngày bắt đầu (YYYY-MM-DD)")
    pipeline_parser.add_argument("--end-date", type=str, help="Ngày kết thúc (YYYY-MM-DD)")
    pipeline_parser.add_argument("--output-dir", "-o", type=str, help="Thư mục lưu dữ liệu đã xử lý")
    pipeline_parser.add_argument("--pipeline-name", type=str, help="Tên của pipeline xử lý (nếu sử dụng pipeline đã đăng ký)")
    pipeline_parser.add_argument("--no-clean/--clean", dest="no_clean", action="store_true", default=False, help="Bỏ qua bước làm sạch dữ liệu")
    pipeline_parser.add_argument("--no-features/--features", dest="no_features", action="store_true", default=False, help="Bỏ qua bước tạo đặc trưng")
    pipeline_parser.add_argument("--all-indicators/--selected-indicators", dest="all_indicators", action="store_true", default=True, help="Sử dụng tất cả các chỉ báo kỹ thuật có sẵn")
    pipeline_parser.add_argument("--preserve-timestamp/--no-preserve-timestamp", dest="preserve_timestamp", action="store_true", default=True, help="Giữ nguyên timestamp trong quá trình xử lý")
    pipeline_parser.add_argument("--verbose", "-v", action="count", default=0, help="Mức độ chi tiết của log (0-2)")
    pipeline_parser.add_argument("--include-sentiment", action="store_true", help="Bao gồm đặc trưng tâm lý thị trường (Fear & Greed Index)")
    
    process_parser.set_defaults(func=handle_process_command)

    return process_parser

def handle_process_command(args, system):
    """
    Xử lý các lệnh liên quan đến xử lý dữ liệu.
    
    Args:
        args: Đối tượng chứa các tham số dòng lệnh
        system: Instance của AutomatedTradingSystem
        
    Returns:
        int: Mã trạng thái (0 nếu thành công, khác 0 nếu lỗi)
    """
    # Nếu đã có process_command, xử lý trực tiếp thay vì gọi các hàm Click
    if hasattr(args, 'process_command') and args.process_command:
        if args.process_command == 'clean':
            try:
                # Thiết lập mức độ chi tiết
                log_level = _get_log_level(args.verbose if hasattr(args, 'verbose') else 0)
                
                # Tạo data pipeline
                pipeline = DataPipeline(logger=logger)
                
                # Chuẩn bị thư mục đầu vào/ra
                input_dir_path, output_dir_path = _prepare_directories(
                    args.input_dir if hasattr(args, 'input_dir') else None,
                    args.output_dir if hasattr(args, 'output_dir') else None,
                    default_input_subdir="collected"
                )
                
                # Tìm các file dữ liệu
                data_paths = _find_data_files(
                    input_dir_path,
                    args.symbols if hasattr(args, 'symbols') else None,
                    args.timeframes if hasattr(args, 'timeframes') else ['1h'],
                    logger
                )
                
                if not data_paths:
                    logger.error(f"Không tìm thấy file dữ liệu phù hợp trong {input_dir_path}")
                    return 1
                
                # Tải dữ liệu
                loaded_data = {}
                for symbol, path in data_paths.items():
                    logger.info(f"Đang tải dữ liệu từ {path} cho {symbol}...")
                    symbol_data = pipeline.load_data(
                        file_paths=path,
                        file_format=path.suffix[1:] if path.suffix else 'csv'
                    )
                    
                    if symbol_data:
                        if symbol in symbol_data:
                            loaded_data[symbol] = symbol_data[symbol]
                        else:
                            first_symbol = next(iter(symbol_data.keys()))
                            loaded_data[symbol] = symbol_data[first_symbol]
                    else:
                        logger.warning(f"Không thể tải dữ liệu cho {symbol}")
                
                if not loaded_data:
                    logger.error("Không có dữ liệu nào được tải thành công")
                    return 1
                
                # Làm sạch dữ liệu theo loại
                data_type = args.data_type if hasattr(args, 'data_type') else 'ohlcv'
                preserve_timestamp = args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                
                cleaned_data = {}
                if data_type == 'ohlcv' or data_type == 'all':
                    cleaned_data = pipeline.clean_data(
                        loaded_data,
                        clean_ohlcv=True,
                        clean_orderbook=(data_type == 'all'),
                        clean_trades=(data_type == 'all'),
                        preserve_timestamp=preserve_timestamp
                    )
                elif data_type == 'trades':
                    cleaned_data = pipeline.clean_data(
                        loaded_data,
                        clean_ohlcv=False,
                        clean_trades=True,
                        preserve_timestamp=preserve_timestamp
                    )
                elif data_type == 'orderbook':
                    cleaned_data = pipeline.clean_data(
                        loaded_data,
                        clean_ohlcv=False,
                        clean_orderbook=True,
                        preserve_timestamp=preserve_timestamp
                    )
                
                if not cleaned_data:
                    logger.error("Không có dữ liệu nào được làm sạch thành công")
                    return 1
                
                # Lưu dữ liệu đã làm sạch
                saved_paths = pipeline.save_data(
                    cleaned_data,
                    output_dir=output_dir_path,
                    file_format='parquet',
                    include_metadata=True,
                    preserve_timestamp=preserve_timestamp
                )
                
                # Hiển thị kết quả
                if saved_paths:
                    print("\n" + "="*50)
                    print(f"Kết quả làm sạch dữ liệu {data_type}:")
                    for symbol, path in saved_paths.items():
                        print(f"  - {symbol}: {path}")
                    print("="*50 + "\n")
                    return 0
                else:
                    logger.error("Không có dữ liệu nào được lưu")
                    return 1
                    
            except Exception as e:
                logger.error(f"Lỗi khi làm sạch dữ liệu: {str(e)}", exc_info=True)
                return 1
                
        elif args.process_command == 'features':
            try:
                # Thiết lập mức độ chi tiết
                log_level = _get_log_level(args.verbose if hasattr(args, 'verbose') else 0)
                
                # Tạo data pipeline
                pipeline = DataPipeline(logger=logger)
                
                # Chuẩn bị thư mục đầu vào/ra
                default_input = "processed" if not (hasattr(args, 'input_dir') and args.input_dir) else None
                input_dir_path, output_dir_path = _prepare_directories(
                    args.input_dir if hasattr(args, 'input_dir') else None,
                    args.output_dir if hasattr(args, 'output_dir') else None,
                    default_input_subdir=default_input
                )
                
                # Nếu thư mục processed không tồn tại, thử sử dụng thư mục collected
                if not input_dir_path.exists():
                    input_dir_path, _ = _prepare_directories(None, None, default_input_subdir="collected")
                    logger.info(f"Thư mục processed không tồn tại, sử dụng thư mục {input_dir_path}")
                
                # Tìm các file dữ liệu
                data_paths = _find_data_files(
                    input_dir_path,
                    args.symbols if hasattr(args, 'symbols') else None,
                    [''],  # Không filter theo timeframe
                    logger
                )
                
                # Xử lý dữ liệu tương tự như trong hàm create_features
                if not data_paths:
                    logger.error(f"Không tìm thấy file dữ liệu phù hợp trong {input_dir_path}")
                    return 1
                
                # Tải dữ liệu
                loaded_data = {}
                for symbol, path in data_paths.items():
                    symbol_data = pipeline.load_data(
                        file_paths=path,
                        file_format=path.suffix[1:] if path.suffix else 'csv'
                    )
                    
                    if symbol_data:
                        if symbol in symbol_data:
                            loaded_data[symbol] = symbol_data[symbol]
                        else:
                            first_symbol = next(iter(symbol_data.keys()))
                            loaded_data[symbol] = symbol_data[first_symbol]
                
                if not loaded_data:
                    logger.error("Không có dữ liệu nào được tải thành công")
                    return 1
                
                # Tạo cấu hình cho việc tạo đặc trưng
                feature_configs = {}
                for symbol in loaded_data.keys():
                    feature_configs[symbol] = {
                        "feature_names": list(args.indicators) if hasattr(args, 'indicators') and args.indicators else None,
                        "generate_labels": args.generate_labels if hasattr(args, 'generate_labels') else True,
                        "label_window": 10,
                        "label_threshold": 0.01
                    }
                
                # Tạo đặc trưng
                featured_data = pipeline.generate_features(
                    loaded_data,
                    feature_configs=feature_configs,
                    all_indicators=args.all_indicators if hasattr(args, 'all_indicators') else False,
                    preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                )
                
                # Bước tiếp theo xử lý và lưu dữ liệu
                # ...còn lại tương tự như trong hàm create_features...
                
                # Hiển thị kết quả cuối cùng
                print("\n" + "="*50)
                print(f"Đã tạo đặc trưng thành công cho {len(featured_data)} cặp tiền")
                print("="*50 + "\n")
                return 0
                
            except Exception as e:
                logger.error(f"Lỗi khi tạo đặc trưng: {str(e)}", exc_info=True)
                return 1
                
        elif args.process_command == 'pipeline':
            try:
                # Thiết lập mức độ chi tiết
                log_level = _get_log_level(args.verbose if hasattr(args, 'verbose') else 0)
                
                # Tạo data pipeline
                pipeline = DataPipeline(logger=logger)
                
                # Chuẩn bị thư mục đầu vào/ra
                input_dir_path, output_dir_path = _prepare_directories(
                    args.input_dir if hasattr(args, 'input_dir') else None,
                    args.output_dir if hasattr(args, 'output_dir') else None,
                    default_input_subdir="collected"
                )
                
                # Chuyển đổi start_date và end_date thành datetime
                start_datetime = None
                end_datetime = None
                
                if hasattr(args, 'start_date') and args.start_date:
                    try:
                        start_datetime = datetime.strptime(args.start_date, "%Y-%m-%d")
                        logger.info(f"Dữ liệu từ ngày: {args.start_date}")
                    except ValueError:
                        logger.error(f"Định dạng ngày không hợp lệ: {args.start_date}, cần định dạng YYYY-MM-DD")
                        return 1
                        
                if hasattr(args, 'end_date') and args.end_date:
                    try:
                        end_datetime = datetime.strptime(args.end_date, "%Y-%m-%d")
                        logger.info(f"Dữ liệu đến ngày: {args.end_date}")
                    except ValueError:
                        logger.error(f"Định dạng ngày không hợp lệ: {args.end_date}, cần định dạng YYYY-MM-DD")
                        return 1
                
                # Kiểm tra xem có pipeline_name
                pipeline_name = args.pipeline_name if hasattr(args, 'pipeline_name') else None
                
                if pipeline_name:
                    # Nếu có pipeline_name, sử dụng run_pipeline trực tiếp
                    logger.info(f"Sử dụng pipeline đã đăng ký: {pipeline_name}")
                    
                    # Kiểm tra xem có symbols
                    if not hasattr(args, 'symbols') or not args.symbols:
                        symbols_list = None
                    else:
                        symbols_list = list(args.symbols)
                    
                    # Chuyển đổi timeframes thành list
                    timeframes_list = list(args.timeframes) if hasattr(args, 'timeframes') and args.timeframes else ["1h"]
                    
                    # Chạy pipeline
                    result_data = asyncio.run(pipeline.run_pipeline(
                        pipeline_name=pipeline_name,
                        input_files=None,
                        exchange_id=None,
                        symbols=symbols_list,
                        timeframe=timeframes_list[0] if timeframes_list else "1h",
                        start_time=start_datetime,
                        end_time=end_datetime,
                        output_dir=output_dir_path,
                        save_results=True,
                        preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                    ))
                    
                    if not result_data:
                        logger.error("Không có kết quả từ pipeline")
                        return 1
                    
                    # Hiển thị kết quả
                    print("\n" + "="*50)
                    print(f"Kết quả xử lý pipeline {pipeline_name}:")
                    for symbol in result_data.keys():
                        print(f"  - {symbol}: {len(result_data[symbol])} dòng, {len(result_data[symbol].columns)} cột")
                    print("="*50 + "\n")
                    
                    return 0
                
                # Nếu không có pipeline_name, thực hiện xử lý theo từng bước
                # Tìm các file dữ liệu
                data_paths = _find_data_files(
                    input_dir_path,
                    args.symbols if hasattr(args, 'symbols') else None,
                    args.timeframes if hasattr(args, 'timeframes') else ["1h"],
                    logger
                )
                
                if not data_paths:
                    logger.error(f"Không tìm thấy file dữ liệu phù hợp trong {input_dir_path}")
                    
                    # Nếu không tìm thấy file, thử thu thập dữ liệu từ API
                    if hasattr(args, 'symbols') and args.symbols and (hasattr(args, 'start_date') and args.start_date):
                        logger.info(f"Không tìm thấy dữ liệu từ files, thử thu thập từ API...")
                        # Xác định exchange_id (mặc định là binance)
                        exchange_id = args.exchange_id if hasattr(args, 'exchange_id') else "binance"
                        
                        # Thu thập dữ liệu
                        collected_data = asyncio.run(pipeline.collect_data(
                            exchange_id=exchange_id,
                            symbols=list(args.symbols),
                            timeframe=args.timeframes[0] if hasattr(args, 'timeframes') and args.timeframes else "1h",
                            start_time=start_datetime,
                            end_time=end_datetime,
                            preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                        ))
                        
                        if collected_data:
                            # Làm sạch và xử lý dữ liệu thu thập được
                            processed_data = pipeline.clean_data(
                                collected_data,
                                clean_ohlcv=True,
                                preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                            )
                            
                            # Tạo đặc trưng
                            processed_data = pipeline.generate_features(
                                processed_data,
                                all_indicators=args.all_indicators if hasattr(args, 'all_indicators') else True,
                                preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                            )
                            
                            # Lưu kết quả
                            saved_paths = pipeline.save_data(
                                processed_data,
                                output_dir=output_dir_path,
                                file_format='parquet',
                                include_metadata=True,
                                preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                            )
                            
                            if saved_paths:
                                print("\n" + "="*50)
                                print(f"Kết quả xử lý pipeline từ dữ liệu API cho {len(saved_paths)} cặp tiền:")
                                for symbol, path in saved_paths.items():
                                    print(f"  - {symbol}: {len(processed_data[symbol])} dòng, {len(processed_data[symbol].columns)} cột")
                                    print(f"    Lưu tại: {path}")
                                print("="*50 + "\n")
                                return 0
                        
                        logger.error("Không thu thập được dữ liệu từ API")
                    return 1
                
                # Tải dữ liệu từ file
                loaded_data = {}
                for symbol, path in data_paths.items():
                    logger.info(f"Đang tải dữ liệu từ {path} cho {symbol}...")
                    symbol_data = pipeline.load_data(
                        file_paths=path,
                        file_format=path.suffix[1:] if path.suffix else 'csv'
                    )
                    
                    if symbol_data:
                        if symbol in symbol_data:
                            loaded_data[symbol] = symbol_data[symbol]
                        else:
                            # Lấy symbol đầu tiên nếu không tìm thấy
                            first_symbol = next(iter(symbol_data.keys()))
                            loaded_data[symbol] = symbol_data[first_symbol]
                            logger.warning(f"Không tìm thấy dữ liệu cho {symbol}, sử dụng {first_symbol} thay thế")
                    else:
                        logger.warning(f"Không thể tải dữ liệu cho {symbol}")
                
                if not loaded_data:
                    logger.error("Không có dữ liệu nào được tải thành công")
                    return 1
                
                # Xử lý dữ liệu theo từng bước
                processed_data = loaded_data
                
                # Làm sạch dữ liệu nếu cần
                no_clean = args.no_clean if hasattr(args, 'no_clean') else False
                if not no_clean:
                    processed_data = pipeline.clean_data(
                        processed_data,
                        clean_ohlcv=True,
                        clean_orderbook=False,
                        clean_trades=False,
                        preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                    )
                    
                    if not processed_data:
                        logger.error("Không có dữ liệu nào được làm sạch thành công")
                        return 1
                    
                    logger.info(f"Đã làm sạch dữ liệu cho {len(processed_data)} cặp tiền")
                
                # Tạo đặc trưng nếu cần
                no_features = args.no_features if hasattr(args, 'no_features') else False
                if not no_features:
                    processed_data = pipeline.generate_features(
                        processed_data,
                        all_indicators=args.all_indicators if hasattr(args, 'all_indicators') else True,
                        preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                    )
                    
                    if not processed_data:
                        logger.error("Không có đặc trưng nào được tạo thành công")
                        return 1
                    
                    logger.info(f"Đã tạo đặc trưng cho {len(processed_data)} cặp tiền")
                    
                    # Tạo nhãn
                    processed_data = pipeline.create_target_features(
                        processed_data,
                        price_column="close",
                        target_types=["direction", "return", "volatility"],
                        horizons=[1, 3, 5, 10],
                        threshold=0.001
                    )
                    
                    logger.info(f"Đã tạo nhãn mục tiêu cho {len(processed_data)} cặp tiền")
                
                # Lưu kết quả
                saved_paths = pipeline.save_data(
                    processed_data,
                    output_dir=output_dir_path,
                    file_format='parquet',
                    include_metadata=True,
                    preserve_timestamp=args.preserve_timestamp if hasattr(args, 'preserve_timestamp') else True
                )
                
                # Hiển thị kết quả
                if saved_paths:
                    print("\n" + "="*50)
                    print(f"Kết quả xử lý pipeline cho {len(saved_paths)} cặp tiền:")
                    for symbol, path in saved_paths.items():
                        print(f"  - {symbol}: {len(processed_data[symbol])} dòng, {len(processed_data[symbol].columns)} cột")
                        print(f"    Lưu tại: {path}")
                    print("="*50 + "\n")
                    
                    return 0
                else:
                    logger.error("Không có dữ liệu nào được lưu")
                    return 1
                    
            except Exception as e:
                logger.error(f"Lỗi khi chạy pipeline xử lý dữ liệu: {str(e)}", exc_info=True)
                return 1
    
    # Nếu không có subcommand, hiển thị trợ giúp
    print("Sử dụng: main.py process <lệnh> [các tùy chọn]")
    print("\nCác lệnh:")
    print("  clean      Làm sạch dữ liệu thị trường")
    print("  features   Tạo đặc trưng từ dữ liệu thị trường")
    print("  pipeline   Chạy toàn bộ pipeline xử lý dữ liệu")
    print("\nĐể xem thêm trợ giúp chi tiết về một lệnh, sử dụng:")
    print("  main.py process <lệnh> --help")
    return 0

if __name__ == "__main__":
    # Nếu chạy trực tiếp file này
    process_commands()