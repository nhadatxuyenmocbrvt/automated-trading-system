"""
Module quản lý logging cho hệ thống giao dịch tự động.
File này cung cấp các lớp và hàm cho việc ghi log chi tiết, dễ đọc và có định dạng
nhất quán trong toàn bộ hệ thống, đồng thời hỗ trợ nhiều đầu ra (console, file, v.v).
"""

import os
import json
import logging
import datetime
import threading
import traceback
from typing import Dict, List, Tuple, Any, Optional, Union
from pathlib import Path
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import colorama
from colorama import Fore, Back, Style

# Khởi tạo colorama để hỗ trợ màu trong console
colorama.init()

# Import các module từ hệ thống
from config.system_config import get_system_config, LOG_DIR
from config.logging_config import get_logger as get_config_logger

class LogFormatterStyle:
    """
    Lớp định nghĩa các kiểu định dạng log khác nhau.
    Cung cấp các màu sắc và định dạng cụ thể cho các loại log khác nhau.
    """
    
    # Màu cho các mức log
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.WHITE + Back.RED
    }
    
    # Định dạng thời gian
    DEFAULT_TIME_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    # Định dạng mặc định
    DEFAULT_FORMAT = "%(asctime)s - %(name)s - [%(levelname)s] - %(message)s"
    
    # Định dạng chi tiết (bao gồm file, dòng, thread)
    DETAILED_FORMAT = "%(asctime)s - %(name)s - [%(levelname)s] - [%(filename)s:%(lineno)d] - [Thread:%(threadName)s] - %(message)s"
    
    # Định dạng gọn (chỉ bao gồm thông tin thiết yếu)
    MINIMAL_FORMAT = "[%(levelname)s] - %(message)s"
    
    # Định dạng cho trading logs
    TRADING_FORMAT = "%(asctime)s - [TRADING] - [%(levelname)s] - [%(symbol)s] - %(message)s"
    
    # Định dạng cho training logs
    TRAINING_FORMAT = "%(asctime)s - [TRAINING] - [%(levelname)s] - [EP:%(episode)s] - %(message)s"
    
    # Định dạng cho API logs
    API_FORMAT = "%(asctime)s - [API] - [%(levelname)s] - [%(method)s] %(url)s - %(message)s"
    
    # Định dạng cho system logs
    SYSTEM_FORMAT = "%(asctime)s - [SYSTEM] - [%(levelname)s] - [%(component)s] - %(message)s"

class ColoredLogFormatter(logging.Formatter):
    """
    Formatter với màu sắc cho console output.
    Định dạng log với màu sắc dựa trên mức độ log.
    """
    
    def __init__(self, fmt=None, datefmt=None, style='%'):
        super().__init__(fmt, datefmt, style)
    
    def format(self, record):
        # Sao chép record để tránh thay đổi gốc
        formatted_record = logging.makeLogRecord(record.__dict__)
        
        # Thêm màu cho levelname dựa trên mức độ log
        level_name = formatted_record.levelname
        if level_name in LogFormatterStyle.COLORS:
            colored_level_name = f"{LogFormatterStyle.COLORS[level_name]}{level_name}{Style.RESET_ALL}"
            formatted_record.levelname = colored_level_name
        
        # Định dạng message với màu thích hợp
        if hasattr(record, 'highlight') and record.highlight:
            formatted_record.msg = f"{Fore.WHITE}{Back.BLUE}{record.msg}{Style.RESET_ALL}"
        
        return super().format(formatted_record)

class LoggerFactory:
    """
    Factory tạo và quản lý các logger trong hệ thống.
    Đảm bảo tính nhất quán và tránh tạo logger trùng lặp.
    """
    
    # Singleton instance
    _instance = None
    _lock = threading.Lock()
    
    # Dictionary để lưu trữ các logger đã tạo
    _loggers = {}
    
    @classmethod
    def get_instance(cls):
        """Lấy instance singleton của LoggerFactory."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance
    
    def __init__(self):
        """Khởi tạo LoggerFactory."""
        self.system_config = get_system_config()
        
        # Mức độ log mặc định từ cấu hình
        self.default_log_level = getattr(logging, 
                                        self.system_config.get("log_level", "INFO"), 
                                        logging.INFO)
        
        # Khởi tạo đường dẫn log
        self.log_dir = LOG_DIR
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tạo các thư mục log con
        self.trading_log_dir = self.log_dir / "trading"
        self.training_log_dir = self.log_dir / "training"
        self.api_log_dir = self.log_dir / "api"
        self.system_log_dir = self.log_dir / "system"
        
        for directory in [self.trading_log_dir, self.training_log_dir, 
                         self.api_log_dir, self.system_log_dir]:
            directory.mkdir(exist_ok=True)
        
        # Khởi tạo formatters
        self._init_formatters()
        
        # Tạo handler mặc định
        self._init_default_handlers()
    
    def _init_formatters(self):
        """Khởi tạo các formatter cho các kiểu log khác nhau."""
        self.formatters = {
            'default': logging.Formatter(LogFormatterStyle.DEFAULT_FORMAT, 
                                          LogFormatterStyle.DEFAULT_TIME_FORMAT),
            'detailed': logging.Formatter(LogFormatterStyle.DETAILED_FORMAT, 
                                           LogFormatterStyle.DEFAULT_TIME_FORMAT),
            'minimal': logging.Formatter(LogFormatterStyle.MINIMAL_FORMAT, 
                                          LogFormatterStyle.DEFAULT_TIME_FORMAT),
            'trading': logging.Formatter(LogFormatterStyle.TRADING_FORMAT, 
                                          LogFormatterStyle.DEFAULT_TIME_FORMAT),
            'training': logging.Formatter(LogFormatterStyle.TRAINING_FORMAT, 
                                           LogFormatterStyle.DEFAULT_TIME_FORMAT),
            'api': logging.Formatter(LogFormatterStyle.API_FORMAT, 
                                      LogFormatterStyle.DEFAULT_TIME_FORMAT),
            'system': logging.Formatter(LogFormatterStyle.SYSTEM_FORMAT, 
                                         LogFormatterStyle.DEFAULT_TIME_FORMAT),
            'colored': ColoredLogFormatter(LogFormatterStyle.DEFAULT_FORMAT, 
                                            LogFormatterStyle.DEFAULT_TIME_FORMAT)
        }
    
    def _init_default_handlers(self):
        """Khởi tạo các handler mặc định cho toàn bộ hệ thống."""
        # Handler cho console
        self.console_handler = logging.StreamHandler()
        self.console_handler.setFormatter(self.formatters['colored'])
        self.console_handler.setLevel(self.default_log_level)
        
        # Handler cho file log chung
        main_log_file = self.log_dir / f"main_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        self.file_handler = RotatingFileHandler(
            main_log_file,
            maxBytes=10*1024*1024,  # 10MB
            backupCount=10,
            encoding='utf-8'
        )
        self.file_handler.setFormatter(self.formatters['detailed'])
        self.file_handler.setLevel(self.default_log_level)
        
        # Handler cho file error log
        error_log_file = self.log_dir / f"error_{datetime.datetime.now().strftime('%Y%m%d')}.log"
        self.error_handler = RotatingFileHandler(
            error_log_file,
            maxBytes=5*1024*1024,  # 5MB
            backupCount=10,
            encoding='utf-8'
        )
        self.error_handler.setFormatter(self.formatters['detailed'])
        self.error_handler.setLevel(logging.ERROR)
    
    def get_logger(self, name: str, log_level: Optional[int] = None, 
                  add_console_handler: bool = True, 
                  add_file_handler: bool = True,
                  log_format: str = 'default') -> logging.Logger:
        """
        Lấy logger theo tên.
        
        Args:
            name: Tên của logger
            log_level: Mức độ log (None để sử dụng mặc định)
            add_console_handler: Thêm handler console
            add_file_handler: Thêm handler file
            log_format: Định dạng log ('default', 'detailed', 'minimal', 'trading', 'training', 'api', 'system')
            
        Returns:
            Logger đã được cấu hình
        """
        # Kiểm tra xem logger đã tồn tại chưa
        if name in self._loggers:
            return self._loggers[name]
        
        # Tạo logger mới
        logger = logging.getLogger(name)
        
        # Đặt mức độ log
        if log_level is None:
            log_level = self.default_log_level
        logger.setLevel(log_level)
        
        # Đảm bảo logger không truyền log lên cha
        logger.propagate = False
        
        # Thêm console handler nếu cần
        if add_console_handler:
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(self.formatters.get(log_format, self.formatters['default']))
            console_handler.setLevel(log_level)
            logger.addHandler(console_handler)
        
        # Thêm file handler nếu cần
        if add_file_handler:
            # Xác định đường dẫn file log phù hợp
            log_file = None
            
            if log_format == 'trading':
                log_file = self.trading_log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            elif log_format == 'training':
                log_file = self.training_log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            elif log_format == 'api':
                log_file = self.api_log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            elif log_format == 'system':
                log_file = self.system_log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            else:
                log_file = self.log_dir / f"{name}_{datetime.datetime.now().strftime('%Y%m%d')}.log"
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5,
                encoding='utf-8'
            )
            file_handler.setFormatter(self.formatters.get(log_format, self.formatters['default']))
            file_handler.setLevel(log_level)
            logger.addHandler(file_handler)
            
            # Thêm handler cho error
            logger.addHandler(self.error_handler)
        
        # Lưu trữ logger
        self._loggers[name] = logger
        
        return logger
    
    def get_trading_logger(self, symbol: str) -> logging.Logger:
        """
        Lấy logger dành riêng cho giao dịch theo symbol.
        
        Args:
            symbol: Symbol giao dịch
            
        Returns:
            Logger được cấu hình cho giao dịch
        """
        # Tạo tên logger
        logger_name = f"trading_{symbol.replace('/', '_')}"
        
        # Lấy logger
        logger = self.get_logger(
            name=logger_name,
            log_format='trading',
            add_console_handler=True,
            add_file_handler=True
        )
        
        return logger
    
    def get_training_logger(self, agent_name: str) -> logging.Logger:
        """
        Lấy logger dành riêng cho huấn luyện theo tên agent.
        
        Args:
            agent_name: Tên agent
            
        Returns:
            Logger được cấu hình cho huấn luyện
        """
        # Tạo tên logger
        logger_name = f"training_{agent_name}"
        
        # Lấy logger
        logger = self.get_logger(
            name=logger_name,
            log_format='training',
            add_console_handler=True,
            add_file_handler=True
        )
        
        return logger
    
    def get_api_logger(self) -> logging.Logger:
        """
        Lấy logger dành riêng cho API.
        
        Returns:
            Logger được cấu hình cho API
        """
        # Lấy logger
        logger = self.get_logger(
            name="api",
            log_format='api',
            add_console_handler=True,
            add_file_handler=True
        )
        
        return logger
    
    def get_system_logger(self, component: str) -> logging.Logger:
        """
        Lấy logger dành riêng cho hệ thống theo tên thành phần.
        
        Args:
            component: Tên thành phần hệ thống
            
        Returns:
            Logger được cấu hình cho hệ thống
        """
        # Tạo tên logger
        logger_name = f"system_{component}"
        
        # Lấy logger
        logger = self.get_logger(
            name=logger_name,
            log_format='system',
            add_console_handler=True,
            add_file_handler=True
        )
        
        return logger
    
    def set_global_log_level(self, level: Union[int, str]) -> None:
        """
        Đặt mức độ log cho tất cả các logger.
        
        Args:
            level: Mức độ log (số nguyên hoặc tên)
        """
        # Chuyển đổi tên thành số nguyên nếu cần
        if isinstance(level, str):
            level = getattr(logging, level.upper(), self.default_log_level)
        
        # Cập nhật mức độ log cho các handler mặc định
        self.console_handler.setLevel(level)
        self.file_handler.setLevel(level)
        
        # Cập nhật mức độ log cho tất cả các logger
        for logger_name, logger in self._loggers.items():
            logger.setLevel(level)
            for handler in logger.handlers:
                if not isinstance(handler, logging.handlers.RotatingFileHandler) or \
                   handler is not self.error_handler:
                    handler.setLevel(level)


class TradeLogger:
    """
    Lớp logger chuyên biệt cho giao dịch.
    Cung cấp các phương thức ghi log cụ thể cho các sự kiện giao dịch khác nhau.
    """
    
    def __init__(self, symbol: str):
        """
        Khởi tạo TradeLogger.
        
        Args:
            symbol: Symbol giao dịch
        """
        self.symbol = symbol
        
        # Lấy logger từ factory
        self.logger = LoggerFactory.get_instance().get_trading_logger(symbol)
        
        # Ghi log khởi tạo
        self.logger.info(f"Khởi tạo TradeLogger cho {symbol}", extra={"symbol": symbol})
    
    def log_trade_signal(self, signal_type: str, direction: str, 
                        price: float, details: Optional[Dict[str, Any]] = None) -> None:
        """
        Ghi log tín hiệu giao dịch.
        
        Args:
            signal_type: Loại tín hiệu (entry, exit, stop_loss, take_profit)
            direction: Hướng (long, short)
            price: Giá tín hiệu
            details: Thông tin chi tiết bổ sung
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "signal_type": signal_type,
            "direction": direction,
            "price": price
        }
        
        # Thêm chi tiết nếu có
        if details:
            extra.update(details)
        
        # Tạo thông điệp log
        message = f"Tín hiệu {signal_type.upper()} {direction} tại giá {price}"
        
        # Log với mức độ phù hợp
        self.logger.info(message, extra=extra)
    
    def log_order_created(self, order_id: str, order_type: str, side: str, 
                         quantity: float, price: Optional[float] = None) -> None:
        """
        Ghi log tạo lệnh giao dịch.
        
        Args:
            order_id: ID lệnh
            order_type: Loại lệnh (market, limit, etc.)
            side: Phía lệnh (buy, sell)
            quantity: Số lượng
            price: Giá lệnh (chỉ cho limit orders)
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "order_id": order_id,
            "order_type": order_type,
            "side": side,
            "quantity": quantity,
            "price": price
        }
        
        # Tạo thông điệp log
        if order_type.lower() == "market":
            message = f"Đã tạo lệnh {order_type.upper()} {side.upper()} {quantity}"
        else:
            message = f"Đã tạo lệnh {order_type.upper()} {side.upper()} {quantity} @ {price}"
        
        # Log với mức độ phù hợp
        self.logger.info(message, extra=extra)
    
    def log_order_filled(self, order_id: str, fill_price: float, 
                        fill_quantity: float, fees: Optional[float] = None) -> None:
        """
        Ghi log lệnh giao dịch được thực hiện.
        
        Args:
            order_id: ID lệnh
            fill_price: Giá thực hiện
            fill_quantity: Số lượng thực hiện
            fees: Phí giao dịch
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "order_id": order_id,
            "fill_price": fill_price,
            "fill_quantity": fill_quantity,
            "fees": fees
        }
        
        # Tạo thông điệp log
        message = f"Lệnh {order_id} đã được thực hiện: {fill_quantity} @ {fill_price}"
        if fees is not None:
            message += f", phí: {fees}"
        
        # Log với mức độ phù hợp
        self.logger.info(message, extra=extra)
    
    def log_order_canceled(self, order_id: str, reason: Optional[str] = None) -> None:
        """
        Ghi log lệnh giao dịch bị hủy.
        
        Args:
            order_id: ID lệnh
            reason: Lý do hủy lệnh
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "order_id": order_id,
            "reason": reason
        }
        
        # Tạo thông điệp log
        message = f"Lệnh {order_id} đã bị hủy"
        if reason:
            message += f": {reason}"
        
        # Log với mức độ phù hợp
        self.logger.info(message, extra=extra)
    
    def log_position_opened(self, position_id: str, side: str, entry_price: float, 
                          size: float, leverage: Optional[float] = None) -> None:
        """
        Ghi log mở vị thế.
        
        Args:
            position_id: ID vị thế
            side: Phía vị thế (long, short)
            entry_price: Giá vào
            size: Kích thước vị thế
            leverage: Đòn bẩy
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "position_id": position_id,
            "side": side,
            "entry_price": entry_price,
            "size": size,
            "leverage": leverage
        }
        
        # Tạo thông điệp log
        message = f"Đã mở vị thế {side.upper()} {size} @ {entry_price}"
        if leverage is not None and leverage > 1.0:
            message += f", đòn bẩy: {leverage}x"
        
        # Log với mức độ phù hợp
        self.logger.info(message, extra=extra, exc_info=None)
    
    def log_position_closed(self, position_id: str, exit_price: float, 
                          profit_loss: float, profit_loss_percent: float,
                          reason: Optional[str] = None) -> None:
        """
        Ghi log đóng vị thế.
        
        Args:
            position_id: ID vị thế
            exit_price: Giá thoát
            profit_loss: Lợi nhuận/thua lỗ (tuyệt đối)
            profit_loss_percent: Lợi nhuận/thua lỗ (phần trăm)
            reason: Lý do đóng vị thế
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "position_id": position_id,
            "exit_price": exit_price,
            "profit_loss": profit_loss,
            "profit_loss_percent": profit_loss_percent,
            "reason": reason
        }
        
        # Xác định mức độ log dựa trên profit_loss
        log_level = logging.INFO
        
        # Tạo thông điệp log
        message = f"Đã đóng vị thế tại {exit_price}, P&L: {profit_loss:.2f} ({profit_loss_percent:.2f}%)"
        if reason:
            message += f", lý do: {reason}"
        
        # Dùng highlight cho các giao dịch lãi/lỗ đáng chú ý
        highlight = False
        if abs(profit_loss_percent) > 5.0:
            highlight = True
        
        # Log với mức độ phù hợp
        self.logger.log(log_level, message, extra={**extra, "highlight": highlight})
    
    def log_stop_loss_triggered(self, position_id: str, stop_price: float,
                              original_price: float, loss_percent: float) -> None:
        """
        Ghi log khi stop loss được kích hoạt.
        
        Args:
            position_id: ID vị thế
            stop_price: Giá dừng lỗ
            original_price: Giá ban đầu
            loss_percent: Phần trăm thua lỗ
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "position_id": position_id,
            "stop_price": stop_price,
            "original_price": original_price,
            "loss_percent": loss_percent
        }
        
        # Tạo thông điệp log
        message = f"Đã kích hoạt dừng lỗ tại {stop_price}, lỗ: {loss_percent:.2f}%"
        
        # Log với mức độ warning
        self.logger.warning(message, extra=extra)
    
    def log_take_profit_triggered(self, position_id: str, take_profit_price: float,
                                original_price: float, profit_percent: float) -> None:
        """
        Ghi log khi take profit được kích hoạt.
        
        Args:
            position_id: ID vị thế
            take_profit_price: Giá chốt lời
            original_price: Giá ban đầu
            profit_percent: Phần trăm lợi nhuận
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "position_id": position_id,
            "take_profit_price": take_profit_price,
            "original_price": original_price,
            "profit_percent": profit_percent
        }
        
        # Tạo thông điệp log
        message = f"Đã kích hoạt chốt lời tại {take_profit_price}, lãi: {profit_percent:.2f}%"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra, exc_info=None)
    
    def log_error(self, error_msg: str, error_code: Optional[int] = None,
                exception: Optional[Exception] = None) -> None:
        """
        Ghi log lỗi giao dịch.
        
        Args:
            error_msg: Thông điệp lỗi
            error_code: Mã lỗi
            exception: Đối tượng exception
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "error_code": error_code
        }
        
        # Tạo thông điệp log
        message = f"Lỗi giao dịch: {error_msg}"
        if error_code is not None:
            message += f" (Mã: {error_code})"
        
        # Log với mức độ error
        if exception is not None:
            self.logger.error(message, extra=extra, exc_info=exception)
        else:
            self.logger.error(message, extra=extra)
    
    def log_portfolio_update(self, total_value: float, positions: Dict[str, Any],
                           available_balance: float) -> None:
        """
        Ghi log cập nhật danh mục đầu tư.
        
        Args:
            total_value: Tổng giá trị danh mục
            positions: Các vị thế hiện tại
            available_balance: Số dư khả dụng
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "total_value": total_value,
            "available_balance": available_balance,
            "position_count": len(positions)
        }
        
        # Tạo thông điệp log
        message = f"Cập nhật danh mục: Tổng giá trị: {total_value:.2f}, Số dư khả dụng: {available_balance:.2f}, Số vị thế: {len(positions)}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_strategy_signal(self, strategy_name: str, signal_type: str,
                          confidence: float, parameters: Dict[str, Any]) -> None:
        """
        Ghi log tín hiệu từ chiến lược giao dịch.
        
        Args:
            strategy_name: Tên chiến lược
            signal_type: Loại tín hiệu (buy, sell, hold)
            confidence: Độ tin cậy của tín hiệu (0.0-1.0)
            parameters: Các tham số bổ sung
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "strategy_name": strategy_name,
            "signal_type": signal_type,
            "confidence": confidence
        }
        
        # Thêm parameters vào extra
        for key, value in parameters.items():
            extra[key] = value
        
        # Tạo thông điệp log
        message = f"Tín hiệu từ chiến lược {strategy_name}: {signal_type.upper()}, độ tin cậy: {confidence:.2f}"
        
        # Log với mức độ phù hợp
        log_level = logging.INFO
        if confidence < 0.5:
            log_level = logging.DEBUG
        
        self.logger.log(log_level, message, extra=extra)
    
    def log_risk_management(self, action: str, value: float, 
                          original_value: Optional[float] = None) -> None:
        """
        Ghi log hành động quản lý rủi ro.
        
        Args:
            action: Hành động (position_size, stop_loss, take_profit)
            value: Giá trị mới
            original_value: Giá trị ban đầu
        """
        # Tạo thông tin bổ sung
        extra = {
            "symbol": self.symbol,
            "action": action,
            "value": value,
            "original_value": original_value
        }
        
        # Tạo thông điệp log
        message = f"Quản lý rủi ro: {action} = {value}"
        if original_value is not None:
            message += f" (từ {original_value})"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)

class SystemLogger:
    """
    Lớp logger chuyên biệt cho hệ thống.
    Cung cấp các phương thức ghi log cụ thể cho các sự kiện hệ thống khác nhau.
    """
    
    def __init__(self, component: str):
        """
        Khởi tạo SystemLogger.
        
        Args:
            component: Tên thành phần hệ thống
        """
        self.component = component
        
        # Lấy logger từ factory
        self.logger = LoggerFactory.get_instance().get_system_logger(component)
        
        # Ghi log khởi tạo
        self.logger.info(f"Khởi tạo SystemLogger cho {component}", extra={"component": component})
    
    def log_startup(self, config: Dict[str, Any]) -> None:
        """
        Ghi log khởi động hệ thống/thành phần.
        
        Args:
            config: Cấu hình khởi động
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            "config": config
        }
        
        # Tạo thông điệp log
        message = f"Khởi động {self.component} với cấu hình: {json.dumps(config, ensure_ascii=False)}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_shutdown(self, reason: Optional[str] = None) -> None:
        """
        Ghi log tắt hệ thống/thành phần.
        
        Args:
            reason: Lý do tắt
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            "reason": reason
        }
        
        # Tạo thông điệp log
        message = f"Tắt {self.component}"
        if reason:
            message += f": {reason}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_error(self, error_msg: str, error_code: Optional[int] = None,
                exception: Optional[Exception] = None) -> None:
        """
        Ghi log lỗi hệ thống.
        
        Args:
            error_msg: Thông điệp lỗi
            error_code: Mã lỗi
            exception: Đối tượng exception
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            "error_code": error_code
        }
        
        # Tạo thông điệp log
        message = f"Lỗi hệ thống: {error_msg}"
        if error_code is not None:
            message += f" (Mã: {error_code})"
        
        # Log với mức độ error
        if exception is not None:
            self.logger.error(message, extra=extra, exc_info=exception)
        else:
            self.logger.error(message, extra=extra)
    
    def log_critical(self, error_msg: str, error_code: Optional[int] = None,
                   exception: Optional[Exception] = None) -> None:
        """
        Ghi log lỗi nghiêm trọng.
        
        Args:
            error_msg: Thông điệp lỗi
            error_code: Mã lỗi
            exception: Đối tượng exception
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            "error_code": error_code
        }
        
        # Tạo thông điệp log
        message = f"LỖI NGHIÊM TRỌNG: {error_msg}"
        if error_code is not None:
            message += f" (Mã: {error_code})"
        
        # Lấy thông tin stacktrace nếu có
        if exception is not None:
            stack_trace = traceback.format_exc()
            message += f"\n{stack_trace}"
        
        # Log với mức độ critical
        self.logger.critical(message, extra=extra, exc_info=True)
    
    def log_config_change(self, param_name: str, old_value: Any, new_value: Any) -> None:
        """
        Ghi log thay đổi cấu hình.
        
        Args:
            param_name: Tên tham số
            old_value: Giá trị cũ
            new_value: Giá trị mới
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            "param_name": param_name,
            "old_value": old_value,
            "new_value": new_value
        }
        
        # Tạo thông điệp log
        message = f"Thay đổi cấu hình: {param_name} từ {old_value} thành {new_value}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_performance(self, metrics: Dict[str, Any]) -> None:
        """
        Ghi log hiệu suất hệ thống.
        
        Args:
            metrics: Các chỉ số hiệu suất
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            **metrics
        }
        
        # Tạo thông điệp log
        message = f"Hiệu suất hệ thống: {json.dumps(metrics, ensure_ascii=False)}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_event(self, event_type: str, event_data: Dict[str, Any]) -> None:
        """
        Ghi log sự kiện hệ thống chung.
        
        Args:
            event_type: Loại sự kiện
            event_data: Dữ liệu sự kiện
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            "event_type": event_type,
            **event_data
        }
        
        # Tạo thông điệp log
        message = f"Sự kiện hệ thống: {event_type} - {json.dumps(event_data, ensure_ascii=False)}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_warning(self, warning_msg: str, suggestion: Optional[str] = None) -> None:
        """
        Ghi log cảnh báo hệ thống.
        
        Args:
            warning_msg: Thông điệp cảnh báo
            suggestion: Gợi ý khắc phục
        """
        # Tạo thông tin bổ sung
        extra = {
            "component": self.component,
            "suggestion": suggestion
        }
        
        # Tạo thông điệp log
        message = f"Cảnh báo: {warning_msg}"
        if suggestion:
            message += f" - Gợi ý: {suggestion}"
        
        # Log với mức độ warning
        self.logger.warning(message, extra=extra)

class TrainingLogger:
    """
    Lớp logger chuyên biệt cho huấn luyện.
    Cung cấp các phương thức ghi log cụ thể cho quá trình huấn luyện agent.
    """
    
    def __init__(self, agent_name: str):
        """
        Khởi tạo TrainingLogger.
        
        Args:
            agent_name: Tên agent
        """
        self.agent_name = agent_name
        
        # Lấy logger từ factory
        self.logger = LoggerFactory.get_instance().get_training_logger(agent_name)
        
        # Ghi log khởi tạo
        self.logger.info(f"Khởi tạo TrainingLogger cho {agent_name}", extra={"agent": agent_name, "episode": 0})
    
    def log_episode_start(self, episode: int, total_episodes: int) -> None:
        """
        Ghi log bắt đầu episode huấn luyện.
        
        Args:
            episode: Số episode hiện tại
            total_episodes: Tổng số episode
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            "total_episodes": total_episodes
        }
        
        # Tạo thông điệp log
        message = f"Bắt đầu episode {episode}/{total_episodes}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_episode_end(self, episode: int, total_episodes: int, reward: float,
                      steps: int, time_taken: float) -> None:
        """
        Ghi log kết thúc episode huấn luyện.
        
        Args:
            episode: Số episode hiện tại
            total_episodes: Tổng số episode
            reward: Phần thưởng total
            steps: Số bước trong episode
            time_taken: Thời gian thực hiện (giây)
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            "total_episodes": total_episodes,
            "reward": reward,
            "steps": steps,
            "time": time_taken
        }
        
        # Tạo thông điệp log
        message = f"Kết thúc episode {episode}/{total_episodes} - Reward: {reward:.2f}, Steps: {steps}, Time: {time_taken:.2f}s"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_training_metrics(self, episode: int, metrics: Dict[str, float]) -> None:
        """
        Ghi log các chỉ số huấn luyện.
        
        Args:
            episode: Số episode hiện tại
            metrics: Các chỉ số huấn luyện
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            **metrics
        }
        
        # Tạo thông điệp log
        message = "Metrics: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Ghi log các siêu tham số huấn luyện.
        
        Args:
            hyperparams: Các siêu tham số
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": 0,
            **hyperparams
        }
        
        # Tạo thông điệp log
        message = f"Hyperparameters: {json.dumps(hyperparams, ensure_ascii=False)}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_validation(self, episode: int, metrics: Dict[str, float]) -> None:
        """
        Ghi log kết quả đánh giá validation.
        
        Args:
            episode: Số episode hiện tại
            metrics: Các chỉ số đánh giá
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            **metrics
        }
        
        # Tạo thông điệp log
        message = "Validation: " + " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_checkpoint(self, episode: int, checkpoint_path: str) -> None:
        """
        Ghi log lưu checkpoint.
        
        Args:
            episode: Số episode hiện tại
            checkpoint_path: Đường dẫn checkpoint
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            "checkpoint_path": checkpoint_path
        }
        
        # Tạo thông điệp log
        message = f"Đã lưu checkpoint tại {checkpoint_path}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_best_model(self, episode: int, model_path: str, reward: float) -> None:
        """
        Ghi log mô hình tốt nhất.
        
        Args:
            episode: Số episode hiện tại
            model_path: Đường dẫn mô hình
            reward: Phần thưởng đạt được
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            "model_path": model_path,
            "reward": reward
        }
        
        # Tạo thông điệp log với highlight
        message = f"MÔ HÌNH TỐT NHẤT tại episode {episode}, reward: {reward:.2f}, được lưu tại {model_path}"
        
        # Log với mức độ info và highlight
        self.logger.info(message, extra={**extra, "highlight": True})
    
    def log_early_stopping(self, episode: int, patience: int, best_reward: float) -> None:
        """
        Ghi log dừng sớm.
        
        Args:
            episode: Số episode hiện tại
            patience: Số lần chờ
            best_reward: Phần thưởng tốt nhất
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            "patience": patience,
            "best_reward": best_reward
        }
        
        # Tạo thông điệp log
        message = f"Dừng sớm tại episode {episode} sau {patience} lần không cải thiện. Best reward: {best_reward:.2f}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_training_error(self, episode: int, error_msg: str, 
                         exception: Optional[Exception] = None) -> None:
        """
        Ghi log lỗi huấn luyện.
        
        Args:
            episode: Số episode hiện tại
            error_msg: Thông điệp lỗi
            exception: Đối tượng exception
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode
        }
        
        # Tạo thông điệp log
        message = f"Lỗi huấn luyện tại episode {episode}: {error_msg}"
        
        # Log với mức độ error
        if exception is not None:
            self.logger.error(message, extra=extra, exc_info=exception)
        else:
            self.logger.error(message, extra=extra)
    
    def log_environment_info(self, env_name: str, state_dim: Any, action_dim: Any) -> None:
        """
        Ghi log thông tin môi trường huấn luyện.
        
        Args:
            env_name: Tên môi trường
            state_dim: Kích thước không gian trạng thái
            action_dim: Kích thước không gian hành động
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": 0,
            "env_name": env_name,
            "state_dim": state_dim,
            "action_dim": action_dim
        }
        
        # Tạo thông điệp log
        message = f"Môi trường huấn luyện: {env_name}, State dim: {state_dim}, Action dim: {action_dim}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)
    
    def log_custom_event(self, event_name: str, episode: int, data: Dict[str, Any]) -> None:
        """
        Ghi log sự kiện tùy chỉnh.
        
        Args:
            event_name: Tên sự kiện
            episode: Số episode hiện tại
            data: Dữ liệu sự kiện
        """
        # Tạo thông tin bổ sung
        extra = {
            "agent": self.agent_name,
            "episode": episode,
            "event": event_name,
            **data
        }
        
        # Tạo thông điệp log
        message = f"Sự kiện {event_name} tại episode {episode}: {json.dumps(data, ensure_ascii=False)}"
        
        # Log với mức độ info
        self.logger.info(message, extra=extra)

# Hàm helper để lấy logger
def get_logger(name: str, log_level: Optional[int] = None) -> logging.Logger:
    """
    Hàm helper để lấy logger thông thường.
    
    Args:
        name: Tên logger
        log_level: Mức độ log
    
    Returns:
        Logger đã được cấu hình
    """
    # Trước tiên, thử sử dụng logger từ config_logging để duy trì tính tương thích
    try:
        return get_config_logger(name)
    except Exception:
        # Nếu không thành công, sử dụng LoggerFactory
        return LoggerFactory.get_instance().get_logger(name, log_level)

def get_trade_logger(symbol: str) -> TradeLogger:
    """
    Hàm helper để lấy logger giao dịch.
    
    Args:
        symbol: Symbol giao dịch
    
    Returns:
        TradeLogger đã được cấu hình
    """
    return TradeLogger(symbol)

def get_system_logger(component: str) -> SystemLogger:
    """
    Hàm helper để lấy logger hệ thống.
    
    Args:
        component: Tên thành phần hệ thống
    
    Returns:
        SystemLogger đã được cấu hình
    """
    return SystemLogger(component)

def get_training_logger(agent_name: str) -> TrainingLogger:
    """
    Hàm helper để lấy logger huấn luyện.
    
    Args:
        agent_name: Tên agent
    
    Returns:
        TrainingLogger đã được cấu hình
    """
    return TrainingLogger(agent_name)