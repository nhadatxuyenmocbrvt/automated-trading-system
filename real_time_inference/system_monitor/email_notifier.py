"""
Thông báo qua email.
File này cung cấp các lớp và hàm để gửi thông báo qua email,
bao gồm cả thông báo thông thường và cảnh báo khi có sự cố.
"""

import os
import smtplib
import ssl
import re
import time
import logging
import threading
import asyncio
import json
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set
from pathlib import Path

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.security_config import get_security_config
from real_time_inference.system_monitor.alert_system import Alert, AlertLevel, AlertCategory
from logs.logger import SystemLogger

class EmailTemplate:
    """
    Lớp quản lý mẫu email.
    Cung cấp các phương thức để tải, chỉnh sửa và áp dụng mẫu email.
    """
    
    def __init__(self, template_content: str = "", template_path: Optional[Union[str, Path]] = None):
        """
        Khởi tạo mẫu email.
        
        Args:
            template_content: Nội dung mẫu
            template_path: Đường dẫn file mẫu (ưu tiên hơn template_content nếu được cung cấp)
        """
        self.logger = SystemLogger("email_template")
        
        if template_path:
            self.load_from_file(template_path)
        else:
            self.template = template_content
    
    def load_from_file(self, file_path: Union[str, Path]) -> bool:
        """
        Tải mẫu từ file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu tải thành công, False nếu không
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                self.template = f.read()
            self.logger.debug(f"Đã tải mẫu email từ {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mẫu email: {str(e)}", exc_info=True)
            self.template = ""
            return False
    
    def save_to_file(self, file_path: Union[str, Path]) -> bool:
        """
        Lưu mẫu vào file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        try:
            # Đảm bảo thư mục tồn tại
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(self.template)
            self.logger.debug(f"Đã lưu mẫu email vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu mẫu email: {str(e)}", exc_info=True)
            return False
    
    def apply(self, parameters: Dict[str, Any]) -> str:
        """
        Áp dụng tham số vào mẫu.
        
        Args:
            parameters: Dict chứa các tham số cần thay thế
            
        Returns:
            Nội dung email sau khi đã thay thế tham số
        """
        if not self.template:
            return ""
        
        content = self.template
        
        # Thay thế các tham số trong mẫu
        for key, value in parameters.items():
            placeholder = f"{{{{{key}}}}}"
            content = content.replace(placeholder, str(value))
        
        return content
    
    def set_template(self, content: str) -> None:
        """
        Đặt nội dung mẫu.
        
        Args:
            content: Nội dung mẫu mới
        """
        self.template = content
    
    def get_template(self) -> str:
        """
        Lấy nội dung mẫu.
        
        Returns:
            Nội dung mẫu
        """
        return self.template

class EmailAccount:
    """
    Lớp quản lý tài khoản email.
    Lưu trữ thông tin đăng nhập và cài đặt SMTP.
    """
    
    def __init__(
        self,
        email: str,
        password: str,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        use_tls: bool = True,
        sender_name: Optional[str] = None
    ):
        """
        Khởi tạo tài khoản email.
        
        Args:
            email: Địa chỉ email
            password: Mật khẩu hoặc mật khẩu ứng dụng
            smtp_server: Địa chỉ server SMTP
            smtp_port: Cổng SMTP
            use_tls: Sử dụng TLS hay không
            sender_name: Tên người gửi (None để sử dụng địa chỉ email)
        """
        self.email = email
        self.password = password
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.use_tls = use_tls
        self.sender_name = sender_name or email
    
    def get_sender(self) -> str:
        """
        Lấy địa chỉ người gửi đầy đủ.
        
        Returns:
            Địa chỉ người gửi đầy đủ (Tên <email>)
        """
        return f"{self.sender_name} <{self.email}>"
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành dictionary.
        
        Returns:
            Dict chứa thông tin tài khoản
        """
        return {
            "email": self.email,
            "password": "********",  # Không lưu mật khẩu thật
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port,
            "use_tls": self.use_tls,
            "sender_name": self.sender_name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], password: Optional[str] = None) -> 'EmailAccount':
        """
        Tạo đối tượng EmailAccount từ dictionary.
        
        Args:
            data: Dictionary chứa thông tin tài khoản
            password: Mật khẩu (None để sử dụng trong data)
            
        Returns:
            Đối tượng EmailAccount
        """
        # Sử dụng mật khẩu được cung cấp hoặc từ data
        actual_password = password or data.get("password", "")
        
        return cls(
            email=data["email"],
            password=actual_password,
            smtp_server=data.get("smtp_server", "smtp.gmail.com"),
            smtp_port=data.get("smtp_port", 587),
            use_tls=data.get("use_tls", True),
            sender_name=data.get("sender_name")
        )

class EmailConfiguration:
    """
    Lớp quản lý cấu hình email.
    Lưu trữ các thiết lập email, mẫu và danh sách người nhận.
    """
    
    def __init__(
        self,
        account: Optional[EmailAccount] = None,
        default_recipients: Optional[List[str]] = None,
        alert_recipients: Optional[Dict[str, List[str]]] = None,
        templates_dir: Optional[Union[str, Path]] = None,
        max_retries: int = 3,
        retry_delay: int = 5,
        throttle_limit: int = 20,
        throttle_period: int = 3600
    ):
        """
        Khởi tạo cấu hình email.
        
        Args:
            account: Tài khoản email
            default_recipients: Danh sách người nhận mặc định
            alert_recipients: Dict với key là tên nhóm cảnh báo, value là danh sách người nhận
            templates_dir: Thư mục chứa các mẫu email
            max_retries: Số lần thử lại tối đa khi gửi thất bại
            retry_delay: Thời gian chờ giữa các lần thử lại (giây)
            throttle_limit: Số email tối đa có thể gửi trong throttle_period
            throttle_period: Khoảng thời gian giới hạn gửi (giây)
        """
        self.logger = SystemLogger("email_configuration")
        
        # Tài khoản email
        self.account = account
        
        # Danh sách người nhận
        self.default_recipients = default_recipients or []
        self.alert_recipients = alert_recipients or {
            "system": [],
            "trading": [],
            "security": [],
            "critical": []
        }
        
        # Cài đặt mẫu
        if templates_dir is None:
            templates_dir = Path("config/templates/email")
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(parents=True, exist_ok=True)
        
        # Cài đặt gửi email
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.throttle_limit = throttle_limit
        self.throttle_period = throttle_period
        
        # Số lượng email đã gửi
        self.sent_timestamps = []
        
        # Mẫu email mặc định
        self.default_templates = {
            "alert": EmailTemplate(),
            "report": EmailTemplate(),
            "notification": EmailTemplate(),
            "welcome": EmailTemplate()
        }
        
        # Tải mẫu mặc định
        self._load_default_templates()
    
    def _load_default_templates(self) -> None:
        """
        Tải các mẫu email mặc định.
        """
        # Mẫu cảnh báo mặc định
        alert_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; color: #333; }
                .container { width: 100%; max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background-color: {{header_color}}; color: white; padding: 10px 20px; }
                .content { padding: 20px; }
                .footer { background-color: #f4f4f4; padding: 10px 20px; font-size: 12px; color: #777; }
                .alert-info { margin: 20px 0; padding: 15px; border-left: 4px solid {{header_color}}; background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{{alert_title}}</h2>
                </div>
                <div class="content">
                    <p>Kính gửi,</p>
                    <p>{{alert_message}}</p>
                    
                    <div class="alert-info">
                        <p><strong>Thời gian:</strong> {{alert_time}}</p>
                        <p><strong>Mức độ:</strong> {{alert_level}}</p>
                        <p><strong>Loại:</strong> {{alert_category}}</p>
                        <p><strong>Nguồn:</strong> {{alert_source}}</p>
                    </div>
                    
                    <p>{{additional_info}}</p>
                    
                    <p>Vui lòng kiểm tra hệ thống và thực hiện các biện pháp cần thiết.</p>
                    <p>Trân trọng,<br>{{sender_name}}</p>
                </div>
                <div class="footer">
                    <p>Email này được gửi tự động từ hệ thống Automated Trading System. Vui lòng không trả lời email này.</p>
                </div>
            </div>
        </body>
        </html>
        """
        self.default_templates["alert"].set_template(alert_template)
        
        # Mẫu báo cáo mặc định
        report_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; color: #333; }
                .container { width: 100%; max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background-color: #2c3e50; color: white; padding: 10px 20px; }
                .content { padding: 20px; }
                .footer { background-color: #f4f4f4; padding: 10px 20px; font-size: 12px; color: #777; }
                .summary { margin: 20px 0; padding: 15px; border-left: 4px solid #2c3e50; background-color: #f9f9f9; }
                table { width: 100%; border-collapse: collapse; margin: 20px 0; }
                th, td { padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{{report_title}}</h2>
                </div>
                <div class="content">
                    <p>Kính gửi,</p>
                    <p>{{report_message}}</p>
                    
                    <div class="summary">
                        <p><strong>Thời gian:</strong> {{report_time}}</p>
                        <p><strong>Giai đoạn báo cáo:</strong> {{report_period}}</p>
                    </div>
                    
                    {{report_content}}
                    
                    <p>Trân trọng,<br>{{sender_name}}</p>
                </div>
                <div class="footer">
                    <p>Email này được gửi tự động từ hệ thống Automated Trading System. Vui lòng không trả lời email này.</p>
                </div>
            </div>
        </body>
        </html>
        """
        self.default_templates["report"].set_template(report_template)
        
        # Mẫu thông báo mặc định
        notification_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; color: #333; }
                .container { width: 100%; max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background-color: #3498db; color: white; padding: 10px 20px; }
                .content { padding: 20px; }
                .footer { background-color: #f4f4f4; padding: 10px 20px; font-size: 12px; color: #777; }
                .notification { margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>{{notification_title}}</h2>
                </div>
                <div class="content">
                    <p>Kính gửi,</p>
                    <p>{{notification_message}}</p>
                    
                    <div class="notification">
                        <p><strong>Thời gian:</strong> {{notification_time}}</p>
                        <p><strong>Loại thông báo:</strong> {{notification_type}}</p>
                    </div>
                    
                    {{notification_content}}
                    
                    <p>Trân trọng,<br>{{sender_name}}</p>
                </div>
                <div class="footer">
                    <p>Email này được gửi tự động từ hệ thống Automated Trading System. Vui lòng không trả lời email này.</p>
                </div>
            </div>
        </body>
        </html>
        """
        self.default_templates["notification"].set_template(notification_template)
        
        # Mẫu chào mừng mặc định
        welcome_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <style>
                body { font-family: Arial, sans-serif; margin: 0; padding: 0; color: #333; }
                .container { width: 100%; max-width: 600px; margin: 0 auto; padding: 20px; }
                .header { background-color: #27ae60; color: white; padding: 10px 20px; }
                .content { padding: 20px; }
                .footer { background-color: #f4f4f4; padding: 10px 20px; font-size: 12px; color: #777; }
                .welcome { margin: 20px 0; padding: 15px; border-left: 4px solid #27ae60; background-color: #f9f9f9; }
                .button { background-color: #27ae60; color: white; padding: 10px 20px; text-decoration: none; display: inline-block; margin: 20px 0; border-radius: 5px; }
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h2>Chào mừng đến với Automated Trading System</h2>
                </div>
                <div class="content">
                    <p>Kính gửi {{user_name}},</p>
                    <p>Chúng tôi rất vui mừng thông báo rằng bạn đã được thêm vào danh sách nhận thông báo từ hệ thống Automated Trading System.</p>
                    
                    <div class="welcome">
                        <p><strong>Tên người dùng:</strong> {{user_name}}</p>
                        <p><strong>Email:</strong> {{user_email}}</p>
                        <p><strong>Đã đăng ký vào:</strong> {{registration_time}}</p>
                        <p><strong>Loại thông báo:</strong> {{notification_types}}</p>
                    </div>
                    
                    <p>Bạn sẽ nhận được thông báo về các sự kiện quan trọng từ hệ thống Automated Trading System.</p>
                    
                    <p>Nếu bạn muốn thay đổi cài đặt thông báo, vui lòng liên hệ với quản trị viên.</p>
                    
                    <p>Trân trọng,<br>{{sender_name}}</p>
                </div>
                <div class="footer">
                    <p>Email này được gửi tự động từ hệ thống Automated Trading System. Vui lòng không trả lời email này.</p>
                </div>
            </div>
        </body>
        </html>
        """
        self.default_templates["welcome"].set_template(welcome_template)
        
        # Lưu các mẫu vào thư mục
        for name, template in self.default_templates.items():
            template_path = self.templates_dir / f"{name}.html"
            if not template_path.exists():
                template.save_to_file(template_path)
    
    def load_template(self, name: str) -> Optional[EmailTemplate]:
        """
        Tải mẫu email từ file.
        
        Args:
            name: Tên mẫu
            
        Returns:
            Đối tượng EmailTemplate hoặc None nếu không tìm thấy
        """
        template_path = self.templates_dir / f"{name}.html"
        
        if not template_path.exists():
            self.logger.warning(f"Không tìm thấy mẫu email '{name}'")
            return None
        
        template = EmailTemplate()
        if template.load_from_file(template_path):
            return template
        
        return None
    
    def get_template(self, name: str) -> EmailTemplate:
        """
        Lấy mẫu email.
        
        Args:
            name: Tên mẫu
            
        Returns:
            Đối tượng EmailTemplate (mẫu mặc định nếu không tìm thấy)
        """
        # Thử tải từ file
        template = self.load_template(name)
        
        # Nếu không tìm thấy, sử dụng mẫu mặc định
        if template is None:
            template = self.default_templates.get(name)
            
            # Nếu không có mẫu mặc định, sử dụng mẫu thông báo
            if template is None:
                template = self.default_templates["notification"]
        
        return template
    
    def add_default_recipient(self, email: str) -> bool:
        """
        Thêm người nhận vào danh sách mặc định.
        
        Args:
            email: Địa chỉ email người nhận
            
        Returns:
            True nếu thêm thành công, False nếu đã tồn tại
        """
        if email in self.default_recipients:
            return False
        
        self.default_recipients.append(email)
        return True
    
    def remove_default_recipient(self, email: str) -> bool:
        """
        Xóa người nhận khỏi danh sách mặc định.
        
        Args:
            email: Địa chỉ email người nhận
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if email in self.default_recipients:
            self.default_recipients.remove(email)
            return True
        
        return False
    
    def add_alert_recipient(self, group: str, email: str) -> bool:
        """
        Thêm người nhận vào nhóm cảnh báo.
        
        Args:
            group: Tên nhóm
            email: Địa chỉ email người nhận
            
        Returns:
            True nếu thêm thành công, False nếu đã tồn tại
        """
        if group not in self.alert_recipients:
            self.alert_recipients[group] = []
        
        if email in self.alert_recipients[group]:
            return False
        
        self.alert_recipients[group].append(email)
        return True
    
    def remove_alert_recipient(self, group: str, email: str) -> bool:
        """
        Xóa người nhận khỏi nhóm cảnh báo.
        
        Args:
            group: Tên nhóm
            email: Địa chỉ email người nhận
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        if group not in self.alert_recipients:
            return False
        
        if email in self.alert_recipients[group]:
            self.alert_recipients[group].remove(email)
            return True
        
        return False
    
    def can_send_email(self) -> bool:
        """
        Kiểm tra xem có thể gửi email hay không (giới hạn tốc độ).
        
        Returns:
            True nếu có thể gửi, False nếu không
        """
        now = time.time()
        
        # Xóa các timestamp cũ
        self.sent_timestamps = [ts for ts in self.sent_timestamps if now - ts < self.throttle_period]
        
        # Kiểm tra giới hạn
        return len(self.sent_timestamps) < self.throttle_limit
    
    def record_email_sent(self) -> None:
        """
        Ghi nhận đã gửi một email.
        """
        self.sent_timestamps.append(time.time())
    
    def save_config(self, file_path: Union[str, Path]) -> bool:
        """
        Lưu cấu hình vào file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            True nếu lưu thành công, False nếu không
        """
        try:
            # Đảm bảo thư mục tồn tại
            Path(file_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Tạo dict cấu hình
            config_dict = {
                "account": self.account.to_dict() if self.account else None,
                "default_recipients": self.default_recipients,
                "alert_recipients": self.alert_recipients,
                "templates_dir": str(self.templates_dir),
                "max_retries": self.max_retries,
                "retry_delay": self.retry_delay,
                "throttle_limit": self.throttle_limit,
                "throttle_period": self.throttle_period
            }
            
            # Lưu vào file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã lưu cấu hình email vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cấu hình email: {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def load_config(cls, file_path: Union[str, Path], password: Optional[str] = None) -> Optional['EmailConfiguration']:
        """
        Tải cấu hình từ file.
        
        Args:
            file_path: Đường dẫn file
            password: Mật khẩu email (None để lấy từ cấu hình)
            
        Returns:
            Đối tượng EmailConfiguration hoặc None nếu không thành công
        """
        logger = SystemLogger("email_configuration")
        
        try:
            # Kiểm tra file tồn tại
            if not Path(file_path).exists():
                logger.error(f"Không tìm thấy file cấu hình email: {file_path}")
                return None
            
            # Đọc cấu hình từ file
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # Tạo đối tượng EmailAccount
            account = None
            if config_dict.get("account"):
                account = EmailAccount.from_dict(config_dict["account"], password)
            
            # Tạo đối tượng EmailConfiguration
            config = cls(
                account=account,
                default_recipients=config_dict.get("default_recipients", []),
                alert_recipients=config_dict.get("alert_recipients", {}),
                templates_dir=config_dict.get("templates_dir"),
                max_retries=config_dict.get("max_retries", 3),
                retry_delay=config_dict.get("retry_delay", 5),
                throttle_limit=config_dict.get("throttle_limit", 20),
                throttle_period=config_dict.get("throttle_period", 3600)
            )
            
            logger.info(f"Đã tải cấu hình email từ {file_path}")
            return config
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình email: {str(e)}", exc_info=True)
            return None

class EmailNotifier:
    """
    Lớp thông báo qua email.
    Quản lý việc gửi email thông báo và cảnh báo.
    """
    
    def __init__(
        self,
        config: Optional[EmailConfiguration] = None,
        config_path: Optional[Union[str, Path]] = None,
        password: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo thông báo email.
        
        Args:
            config: Cấu hình email
            config_path: Đường dẫn file cấu hình (nếu không cung cấp config)
            password: Mật khẩu email (nếu không có trong cấu hình)
            logger: Logger tùy chỉnh
        """
        # Logger
        self.logger = logger or SystemLogger("email_notifier")
        
        # Tải cấu hình
        if config:
            self.config = config
        elif config_path:
            self.config = EmailConfiguration.load_config(config_path, password)
            
            if self.config is None:
                self.logger.warning("Không thể tải cấu hình email, sử dụng cấu hình mặc định")
                self.config = EmailConfiguration()
        else:
            self.config = EmailConfiguration()
        
        # Hàng đợi email
        self.email_queue = asyncio.Queue()
        
        # Thread gửi email
        self.sender_task = None
        self.running = False
        
        # Danh sách email đã gửi gần đây (để khử trùng)
        self.recent_email_hashes = set()
    
    def set_account(
        self,
        email: str,
        password: str,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
        use_tls: bool = True,
        sender_name: Optional[str] = None
    ) -> None:
        """
        Đặt tài khoản email.
        
        Args:
            email: Địa chỉ email
            password: Mật khẩu hoặc mật khẩu ứng dụng
            smtp_server: Địa chỉ server SMTP
            smtp_port: Cổng SMTP
            use_tls: Sử dụng TLS hay không
            sender_name: Tên người gửi (None để sử dụng địa chỉ email)
        """
        self.config.account = EmailAccount(
            email=email,
            password=password,
            smtp_server=smtp_server,
            smtp_port=smtp_port,
            use_tls=use_tls,
            sender_name=sender_name
        )
        
        self.logger.info(f"Đã đặt tài khoản email: {email}")
    
    async def start(self) -> bool:
        """
        Bắt đầu gửi email.
        
        Returns:
            True nếu bắt đầu thành công, False nếu không
        """
        if self.running:
            self.logger.warning("Thông báo email đã đang chạy")
            return False
        
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            self.logger.error("Chưa cấu hình tài khoản email")
            return False
        
        self.running = True
        self.sender_task = asyncio.create_task(self._process_email_queue())
        
        self.logger.info("Đã bắt đầu thông báo email")
        return True
    
    async def stop(self) -> bool:
        """
        Dừng gửi email.
        
        Returns:
            True nếu dừng thành công, False nếu không
        """
        if not self.running:
            self.logger.warning("Thông báo email không đang chạy")
            return False
        
        self.running = False
        
        # Chờ task kết thúc
        if self.sender_task:
            try:
                self.sender_task.cancel()
                await asyncio.wait_for(asyncio.shield(self.sender_task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        self.logger.info("Đã dừng thông báo email")
        return True
    
    async def _process_email_queue(self) -> None:
        """
        Vòng lặp xử lý hàng đợi email.
        """
        self.logger.info("Bắt đầu vòng lặp xử lý email")
        
        while self.running:
            try:
                # Lấy email từ hàng đợi
                email_data = await self.email_queue.get()
                
                # Kiểm tra giới hạn tốc độ
                if not self.config.can_send_email():
                    self.logger.warning("Đã đạt giới hạn gửi email, đợi và thử lại sau")
                    # Đặt lại vào hàng đợi
                    await self.email_queue.put(email_data)
                    # Đợi một chút trước khi thử lại
                    await asyncio.sleep(60)
                    continue
                
                # Gửi email
                success = await self._send_email_with_retry(
                    to=email_data["to"],
                    subject=email_data["subject"],
                    body=email_data["body"],
                    attachments=email_data.get("attachments"),
                    is_html=email_data.get("is_html", True)
                )
                
                # Đánh dấu đã xử lý
                self.email_queue.task_done()
                
                if success:
                    # Ghi nhận đã gửi thành công
                    self.config.record_email_sent()
                else:
                    self.logger.error(f"Không thể gửi email đến {', '.join(email_data['to'])}")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp xử lý email: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Đợi một chút trước khi thử lại
        
        self.logger.info("Kết thúc vòng lặp xử lý email")
    
    async def _send_email_with_retry(
        self,
        to: List[str],
        subject: str,
        body: str,
        attachments: Optional[List[Tuple[str, bytes]]] = None,
        is_html: bool = True
    ) -> bool:
        """
        Gửi email với cơ chế thử lại.
        
        Args:
            to: Danh sách người nhận
            subject: Tiêu đề
            body: Nội dung
            attachments: Danh sách tệp đính kèm (tên file, dữ liệu)
            is_html: True nếu nội dung là HTML
            
        Returns:
            True nếu gửi thành công, False nếu không
        """
        retries = 0
        
        while retries < self.config.max_retries:
            try:
                # Gửi email
                success = await self._send_email(to, subject, body, attachments, is_html)
                
                if success:
                    return True
                
                # Nếu không thành công, thử lại
                retries += 1
                self.logger.warning(f"Gửi email thất bại, thử lại ({retries}/{self.config.max_retries})")
                await asyncio.sleep(self.config.retry_delay)
            except Exception as e:
                retries += 1
                self.logger.error(f"Lỗi khi gửi email: {str(e)}, thử lại ({retries}/{self.config.max_retries})", exc_info=True)
                await asyncio.sleep(self.config.retry_delay)
        
        return False
    
    async def _send_email(
        self,
        to: List[str],
        subject: str,
        body: str,
        attachments: Optional[List[Tuple[str, bytes]]] = None,
        is_html: bool = True
    ) -> bool:
        """
        Gửi email.
        
        Args:
            to: Danh sách người nhận
            subject: Tiêu đề
            body: Nội dung
            attachments: Danh sách tệp đính kèm (tên file, dữ liệu)
            is_html: True nếu nội dung là HTML
            
        Returns:
            True nếu gửi thành công, False nếu không
        """
        # Kiểm tra tham số
        if not to or not subject or not body:
            self.logger.error("Thiếu thông tin để gửi email")
            return False
        
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            self.logger.error("Chưa cấu hình tài khoản email")
            return False
        
        try:
            # Tạo message
            msg = MIMEMultipart("alternative" if is_html else "mixed")
            msg["Subject"] = subject
            msg["From"] = self.config.account.get_sender()
            msg["To"] = ", ".join(to)
            
            # Thêm nội dung
            if is_html:
                msg.attach(MIMEText(body, "html", "utf-8"))
            else:
                msg.attach(MIMEText(body, "plain", "utf-8"))
            
            # Thêm tệp đính kèm
            if attachments:
                for filename, file_data in attachments:
                    part = MIMEApplication(file_data)
                    part.add_header("Content-Disposition", f"attachment; filename={filename}")
                    msg.attach(part)
            
            # Kết nối SMTP
            context = ssl.create_default_context() if self.config.account.use_tls else None
            
            with smtplib.SMTP(self.config.account.smtp_server, self.config.account.smtp_port) as server:
                if self.config.account.use_tls:
                    server.starttls(context=context)
                
                server.login(self.config.account.email, self.config.account.password)
                server.send_message(msg)
            
            self.logger.info(f"Đã gửi email đến {len(to)} người nhận")
            return True
            
        except Exception as e:
            self.logger.error(f"Lỗi khi gửi email: {str(e)}", exc_info=True)
            return False
    
    def validate_email(self, email: str) -> bool:
        """
        Kiểm tra tính hợp lệ của địa chỉ email.
        
        Args:
            email: Địa chỉ email cần kiểm tra
            
        Returns:
            True nếu hợp lệ, False nếu không
        """
        # Sử dụng regex để kiểm tra
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(email_pattern, email) is not None
    
    def _get_recipients_for_alert(self, category: AlertCategory) -> List[str]:
        """
        Lấy danh sách người nhận cho loại cảnh báo.
        
        Args:
            category: Loại cảnh báo
            
        Returns:
            Danh sách người nhận
        """
        recipients = set(self.config.default_recipients)
        
        # Thêm người nhận theo loại cảnh báo
        if category == AlertCategory.SYSTEM:
            recipients.update(self.config.alert_recipients.get("system", []))
        elif category == AlertCategory.TRADING:
            recipients.update(self.config.alert_recipients.get("trading", []))
        elif category == AlertCategory.SECURITY:
            recipients.update(self.config.alert_recipients.get("security", []))
        elif category == AlertCategory.FINANCIAL:
            recipients.update(self.config.alert_recipients.get("financial", []))
        
        # Thêm người nhận cho cảnh báo nghiêm trọng
        recipients.update(self.config.alert_recipients.get("critical", []))
        
        return list(recipients)
    
    def _get_header_color_for_level(self, level: AlertLevel) -> str:
        """
        Lấy mã màu cho header email dựa trên cấp độ cảnh báo.
        
        Args:
            level: Cấp độ cảnh báo
            
        Returns:
            Mã màu HEX
        """
        colors = {
            AlertLevel.INFO: "#3498db",     # Xanh dương
            AlertLevel.NOTICE: "#2ecc71",   # Xanh lá
            AlertLevel.WARNING: "#f39c12",  # Cam
            AlertLevel.ALERT: "#e74c3c",    # Đỏ
            AlertLevel.CRITICAL: "#c0392b"  # Đỏ đậm
        }
        
        return colors.get(level, "#3498db")
    
    def _get_alert_title(self, level: AlertLevel, category: AlertCategory) -> str:
        """
        Tạo tiêu đề cho email cảnh báo.
        
        Args:
            level: Cấp độ cảnh báo
            category: Loại cảnh báo
            
        Returns:
            Tiêu đề cảnh báo
        """
        level_prefix = {
            AlertLevel.INFO: "Thông tin",
            AlertLevel.NOTICE: "Thông báo",
            AlertLevel.WARNING: "Cảnh báo",
            AlertLevel.ALERT: "CẢNH BÁO",
            AlertLevel.CRITICAL: "CẢNH BÁO NGHIÊM TRỌNG"
        }
        
        category_name = {
            AlertCategory.SYSTEM: "Hệ thống",
            AlertCategory.NETWORK: "Mạng",
            AlertCategory.SECURITY: "Bảo mật",
            AlertCategory.TRADING: "Giao dịch",
            AlertCategory.FINANCIAL: "Tài chính",
            AlertCategory.PERFORMANCE: "Hiệu suất",
            AlertCategory.API: "API",
            AlertCategory.DATABASE: "Cơ sở dữ liệu",
            AlertCategory.CUSTOM: "Tùy chỉnh"
        }
        
        prefix = level_prefix.get(level, "Thông báo")
        name = category_name.get(category, "Hệ thống")
        
        return f"{prefix}: {name}"
    
    def _calculate_email_hash(self, subject: str, body: str) -> str:
        """
        Tính toán hash cho email để khử trùng.
        
        Args:
            subject: Tiêu đề email
            body: Nội dung email
            
        Returns:
            Hash email
        """
        import hashlib
        
        # Tạo chuỗi key từ tiêu đề và một phần nội dung
        key = f"{subject}:{body[:100]}"
        
        # Băm chuỗi key
        return hashlib.md5(key.encode()).hexdigest()
    
    async def _remove_email_hash(self, email_hash: str) -> None:
        """
        Xóa hash email sau khoảng thời gian.
        
        Args:
            email_hash: Hash email cần xóa
        """
        await asyncio.sleep(3600)  # 1 giờ
        self.recent_email_hashes.discard(email_hash)
    
    async def send_alert_email(self, alert: Alert) -> bool:
        """
        Gửi email cảnh báo.
        
        Args:
            alert: Đối tượng cảnh báo
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            self.logger.error("Chưa cấu hình tài khoản email")
            return False
        
        # Lấy danh sách người nhận
        recipients = self._get_recipients_for_alert(alert.category)
        
        if not recipients:
            self.logger.warning(f"Không có người nhận cho cảnh báo {alert.alert_id}")
            return False
        
        # Tạo tiêu đề
        subject = self._get_alert_title(alert.level, alert.category)
        
        # Lấy mẫu email
        template = self.config.get_template("alert")
        
        # Tạo tham số
        parameters = {
            "alert_title": subject,
            "alert_message": alert.message,
            "alert_time": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "alert_level": alert.level.name,
            "alert_category": alert.category.name,
            "alert_source": alert.source,
            "header_color": self._get_header_color_for_level(alert.level),
            "additional_info": "",
            "sender_name": self.config.account.sender_name
        }
        
        # Thêm thông tin bổ sung nếu có
        if alert.data:
            additional_info = "<ul>"
            for key, value in alert.data.items():
                additional_info += f"<li><strong>{key}:</strong> {value}</li>"
            additional_info += "</ul>"
            parameters["additional_info"] = additional_info
        
        # Tạo nội dung email
        body = template.apply(parameters)
        
        # Tính hash để khử trùng
        email_hash = self._calculate_email_hash(subject, body)
        
        # Kiểm tra trùng lặp
        if email_hash in self.recent_email_hashes:
            self.logger.debug(f"Bỏ qua email trùng lặp: {subject}")
            return False
        
        # Thêm hash vào danh sách gần đây
        self.recent_email_hashes.add(email_hash)
        
        # Lên lịch xóa hash sau khoảng thời gian
        asyncio.create_task(self._remove_email_hash(email_hash))
        
        # Thêm vào hàng đợi
        await self.email_queue.put({
            "to": recipients,
            "subject": subject,
            "body": body,
            "attachments": None,
            "is_html": True
        })
        
        self.logger.info(f"Đã thêm email cảnh báo vào hàng đợi: {subject}")
        return True
    
    async def send_notification_email(
        self,
        subject: str,
        message: str,
        notification_type: str = "general",
        recipients: Optional[List[str]] = None,
        content: str = "",
        attachments: Optional[List[Tuple[str, bytes]]] = None
    ) -> bool:
        """
        Gửi email thông báo.
        
        Args:
            subject: Tiêu đề
            message: Thông điệp
            notification_type: Loại thông báo
            recipients: Danh sách người nhận (None để sử dụng danh sách mặc định)
            content: Nội dung HTML bổ sung
            attachments: Danh sách tệp đính kèm
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            self.logger.error("Chưa cấu hình tài khoản email")
            return False
        
        # Lấy danh sách người nhận
        if recipients is None:
            recipients = self.config.default_recipients
        
        if not recipients:
            self.logger.warning(f"Không có người nhận cho thông báo: {subject}")
            return False
        
        # Lấy mẫu email
        template = self.config.get_template("notification")
        
        # Tạo tham số
        parameters = {
            "notification_title": subject,
            "notification_message": message,
            "notification_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notification_type": notification_type,
            "notification_content": content,
            "sender_name": self.config.account.sender_name
        }
        
        # Tạo nội dung email
        body = template.apply(parameters)
        
        # Tính hash để khử trùng
        email_hash = self._calculate_email_hash(subject, body)
        
        # Kiểm tra trùng lặp
        if email_hash in self.recent_email_hashes:
            self.logger.debug(f"Bỏ qua email trùng lặp: {subject}")
            return False
        
        # Thêm hash vào danh sách gần đây
        self.recent_email_hashes.add(email_hash)
        
        # Lên lịch xóa hash sau khoảng thời gian
        asyncio.create_task(self._remove_email_hash(email_hash))
        
        # Thêm vào hàng đợi
        await self.email_queue.put({
            "to": recipients,
            "subject": subject,
            "body": body,
            "attachments": attachments,
            "is_html": True
        })
        
        self.logger.info(f"Đã thêm email thông báo vào hàng đợi: {subject}")
        return True
    
    async def send_report_email(
        self,
        title: str,
        message: str,
        report_content: str,
        report_period: str,
        recipients: Optional[List[str]] = None,
        attachments: Optional[List[Tuple[str, bytes]]] = None
    ) -> bool:
        """
        Gửi email báo cáo.
        
        Args:
            title: Tiêu đề báo cáo
            message: Thông điệp
            report_content: Nội dung HTML báo cáo
            report_period: Giai đoạn báo cáo
            recipients: Danh sách người nhận (None để sử dụng danh sách mặc định)
            attachments: Danh sách tệp đính kèm
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            self.logger.error("Chưa cấu hình tài khoản email")
            return False
        
        # Lấy danh sách người nhận
        if recipients is None:
            recipients = self.config.default_recipients
        
        if not recipients:
            self.logger.warning(f"Không có người nhận cho báo cáo: {title}")
            return False
        
        # Lấy mẫu email
        template = self.config.get_template("report")
        
        # Tạo tham số
        parameters = {
            "report_title": title,
            "report_message": message,
            "report_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "report_period": report_period,
            "report_content": report_content,
            "sender_name": self.config.account.sender_name
        }
        
        # Tạo nội dung email
        body = template.apply(parameters)
        
        # Tính hash để khử trùng
        email_hash = self._calculate_email_hash(title, body)
        
        # Kiểm tra trùng lặp
        if email_hash in self.recent_email_hashes:
            self.logger.debug(f"Bỏ qua email trùng lặp: {title}")
            return False
        
        # Thêm hash vào danh sách gần đây
        self.recent_email_hashes.add(email_hash)
        
        # Lên lịch xóa hash sau khoảng thời gian
        asyncio.create_task(self._remove_email_hash(email_hash))
        
        # Thêm vào hàng đợi
        await self.email_queue.put({
            "to": recipients,
            "subject": title,
            "body": body,
            "attachments": attachments,
            "is_html": True
        })
        
        self.logger.info(f"Đã thêm email báo cáo vào hàng đợi: {title}")
        return True
    
    async def send_welcome_email(self, email: str, name: str, notification_types: List[str]) -> bool:
        """
        Gửi email chào mừng.
        
        Args:
            email: Địa chỉ email người nhận
            name: Tên người nhận
            notification_types: Danh sách loại thông báo đã đăng ký
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            self.logger.error("Chưa cấu hình tài khoản email")
            return False
        
        # Kiểm tra địa chỉ email
        if not self.validate_email(email):
            self.logger.error(f"Địa chỉ email không hợp lệ: {email}")
            return False
        
        # Lấy mẫu email
        template = self.config.get_template("welcome")
        
        # Tạo tham số
        parameters = {
            "user_name": name,
            "user_email": email,
            "registration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notification_types": ", ".join(notification_types),
            "sender_name": self.config.account.sender_name
        }
        
        # Tạo nội dung email
        body = template.apply(parameters)
        
        # Thêm vào hàng đợi
        await self.email_queue.put({
            "to": [email],
            "subject": "Chào mừng đến với Automated Trading System",
            "body": body,
            "attachments": None,
            "is_html": True
        })
        
        self.logger.info(f"Đã thêm email chào mừng vào hàng đợi: {email}")
        return True
    
    async def send_custom_email(
        self,
        recipients: List[str],
        subject: str,
        body: str,
        attachments: Optional[List[Tuple[str, bytes]]] = None,
        is_html: bool = True
    ) -> bool:
        """
        Gửi email tùy chỉnh.
        
        Args:
            recipients: Danh sách người nhận
            subject: Tiêu đề
            body: Nội dung
            attachments: Danh sách tệp đính kèm
            is_html: True nếu nội dung là HTML
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            self.logger.error("Chưa cấu hình tài khoản email")
            return False
        
        # Kiểm tra danh sách người nhận
        valid_recipients = [r for r in recipients if self.validate_email(r)]
        
        if not valid_recipients:
            self.logger.error("Không có địa chỉ email hợp lệ trong danh sách người nhận")
            return False
        
        # Tính hash để khử trùng
        email_hash = self._calculate_email_hash(subject, body)
        
        # Kiểm tra trùng lặp
        if email_hash in self.recent_email_hashes:
            self.logger.debug(f"Bỏ qua email trùng lặp: {subject}")
            return False
        
        # Thêm hash vào danh sách gần đây
        self.recent_email_hashes.add(email_hash)
        
        # Lên lịch xóa hash sau khoảng thời gian
        asyncio.create_task(self._remove_email_hash(email_hash))
        
        # Thêm vào hàng đợi
        await self.email_queue.put({
            "to": valid_recipients,
            "subject": subject,
            "body": body,
            "attachments": attachments,
            "is_html": is_html
        })
        
        self.logger.info(f"Đã thêm email tùy chỉnh vào hàng đợi: {subject}")
        return True
    
    def test_connection(self) -> Tuple[bool, Optional[str]]:
        """
        Kiểm tra kết nối đến server SMTP.
        
        Returns:
            Tuple (success, error_message)
        """
        # Kiểm tra cấu hình tài khoản
        if not self.config.account:
            return False, "Chưa cấu hình tài khoản email"
        
        try:
            # Kết nối SMTP
            context = ssl.create_default_context() if self.config.account.use_tls else None
            
            with smtplib.SMTP(self.config.account.smtp_server, self.config.account.smtp_port) as server:
                if self.config.account.use_tls:
                    server.starttls(context=context)
                
                server.login(self.config.account.email, self.config.account.password)
            
            return True, None
            
        except Exception as e:
            return False, str(e)

# Singleton instance
_email_notifier_instance = None

def get_email_notifier() -> EmailNotifier:
    """
    Lấy instance singleton của EmailNotifier.
    
    Returns:
        Instance EmailNotifier
    """
    global _email_notifier_instance
    if _email_notifier_instance is None:
        _email_notifier_instance = EmailNotifier()
    return _email_notifier_instance