"""
Thông báo qua Telegram.
File này cung cấp các lớp và hàm để gửi thông báo qua Telegram,
bao gồm cả thông báo thông thường và cảnh báo khi có sự cố.
"""

import os
import re
import time
import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional, Union, Set, Callable
from pathlib import Path
from io import BytesIO

# Import các module từ hệ thống
from config.logging_config import get_logger
from config.system_config import get_system_config
from config.security_config import get_security_config
from real_time_inference.system_monitor.alert_system import Alert, AlertLevel, AlertCategory
from logs.logger import SystemLogger

class TelegramBot:
    """
    Lớp quản lý bot Telegram.
    Cung cấp các phương thức để gửi tin nhắn và tương tác với API Telegram.
    """
    
    def __init__(
        self,
        token: str,
        base_url: str = "https://api.telegram.org",
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo bot Telegram.
        
        Args:
            token: Token bot Telegram
            base_url: URL cơ sở của API Telegram
            logger: Logger tùy chỉnh
        """
        self.token = token
        self.base_url = base_url.rstrip("/")
        self.api_url = f"{self.base_url}/bot{self.token}"
        self.logger = logger or SystemLogger("telegram_bot")
        
        # Thời gian gửi tin nhắn gần nhất
        self.last_message_time = 0
        
        # HTTP session
        self.session = None
    
    async def create_session(self) -> None:
        """
        Tạo HTTP session mới.
        """
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
    
    async def close_session(self) -> None:
        """
        Đóng HTTP session.
        """
        if self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None
    
    async def get_me(self) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin bot.
        
        Returns:
            Dict thông tin bot hoặc None nếu có lỗi
        """
        await self.create_session()
        
        try:
            url = f"{self.api_url}/getMe"
            async with self.session.get(url) as response:
                result = await response.json()
                
                if result.get("ok"):
                    return result.get("result")
                else:
                    self.logger.error(f"Lỗi khi lấy thông tin bot: {result.get('description')}")
                    return None
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API getMe: {str(e)}", exc_info=True)
            return None
    
    async def get_updates(
        self,
        offset: Optional[int] = None,
        limit: int = 100,
        timeout: int = 30,
        allowed_updates: Optional[List[str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Lấy các cập nhật từ Telegram.
        
        Args:
            offset: ID cập nhật bắt đầu
            limit: Số lượng cập nhật tối đa
            timeout: Thời gian chờ long polling (giây)
            allowed_updates: Loại cập nhật cho phép
            
        Returns:
            Danh sách các cập nhật
        """
        await self.create_session()
        
        try:
            url = f"{self.api_url}/getUpdates"
            params = {"timeout": timeout, "limit": limit}
            
            if offset is not None:
                params["offset"] = offset
            
            if allowed_updates is not None:
                params["allowed_updates"] = json.dumps(allowed_updates)
            
            async with self.session.get(url, params=params) as response:
                result = await response.json()
                
                if result.get("ok"):
                    return result.get("result", [])
                else:
                    self.logger.error(f"Lỗi khi lấy cập nhật: {result.get('description')}")
                    return []
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API getUpdates: {str(e)}", exc_info=True)
            return []
    
    async def send_message(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: Optional[str] = "HTML",
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None,
        disable_web_page_preview: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Gửi tin nhắn văn bản.
        
        Args:
            chat_id: ID chat nhận tin nhắn
            text: Nội dung tin nhắn
            parse_mode: Chế độ phân tích ("HTML", "Markdown" hoặc None)
            disable_notification: Tắt thông báo
            reply_to_message_id: ID tin nhắn trả lời
            disable_web_page_preview: Tắt xem trước trang web
            
        Returns:
            Dict kết quả hoặc None nếu có lỗi
        """
        await self.create_session()
        
        # Đảm bảo không gửi quá nhanh
        current_time = time.time()
        delay = max(0, self.last_message_time + 0.1 - current_time)
        if delay > 0:
            await asyncio.sleep(delay)
        
        try:
            url = f"{self.api_url}/sendMessage"
            payload = {
                "chat_id": chat_id,
                "text": text,
                "disable_web_page_preview": disable_web_page_preview,
                "disable_notification": disable_notification
            }
            
            if parse_mode:
                payload["parse_mode"] = parse_mode
            
            if reply_to_message_id:
                payload["reply_to_message_id"] = reply_to_message_id
            
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                
                if result.get("ok"):
                    self.last_message_time = time.time()
                    return result.get("result")
                else:
                    self.logger.error(f"Lỗi khi gửi tin nhắn: {result.get('description')}")
                    return None
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API sendMessage: {str(e)}", exc_info=True)
            return None
    
    async def send_photo(
        self,
        chat_id: Union[int, str],
        photo: Union[str, bytes, BytesIO],
        caption: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Gửi ảnh.
        
        Args:
            chat_id: ID chat nhận tin nhắn
            photo: URL ảnh hoặc dữ liệu nhị phân
            caption: Chú thích ảnh
            parse_mode: Chế độ phân tích ("HTML", "Markdown" hoặc None)
            disable_notification: Tắt thông báo
            reply_to_message_id: ID tin nhắn trả lời
            
        Returns:
            Dict kết quả hoặc None nếu có lỗi
        """
        await self.create_session()
        
        # Đảm bảo không gửi quá nhanh
        current_time = time.time()
        delay = max(0, self.last_message_time + 0.1 - current_time)
        if delay > 0:
            await asyncio.sleep(delay)
        
        try:
            url = f"{self.api_url}/sendPhoto"
            
            data = aiohttp.FormData()
            data.add_field("chat_id", str(chat_id))
            
            if isinstance(photo, str) and (photo.startswith("http://") or photo.startswith("https://")):
                # Nếu là URL
                data.add_field("photo", photo)
            else:
                # Nếu là dữ liệu nhị phân
                if isinstance(photo, bytes):
                    photo_data = BytesIO(photo)
                else:
                    photo_data = photo
                
                filename = f"photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                data.add_field("photo", photo_data, filename=filename, content_type="image/jpeg")
            
            if caption:
                data.add_field("caption", caption)
            
            if parse_mode:
                data.add_field("parse_mode", parse_mode)
            
            data.add_field("disable_notification", str(disable_notification).lower())
            
            if reply_to_message_id:
                data.add_field("reply_to_message_id", str(reply_to_message_id))
            
            async with self.session.post(url, data=data) as response:
                result = await response.json()
                
                if result.get("ok"):
                    self.last_message_time = time.time()
                    return result.get("result")
                else:
                    self.logger.error(f"Lỗi khi gửi ảnh: {result.get('description')}")
                    return None
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API sendPhoto: {str(e)}", exc_info=True)
            return None
    
    async def send_document(
        self,
        chat_id: Union[int, str],
        document: Union[str, bytes, BytesIO],
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
        disable_notification: bool = False,
        reply_to_message_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Gửi tệp đính kèm.
        
        Args:
            chat_id: ID chat nhận tin nhắn
            document: URL tệp hoặc dữ liệu nhị phân
            filename: Tên tệp
            caption: Chú thích tệp
            parse_mode: Chế độ phân tích ("HTML", "Markdown" hoặc None)
            disable_notification: Tắt thông báo
            reply_to_message_id: ID tin nhắn trả lời
            
        Returns:
            Dict kết quả hoặc None nếu có lỗi
        """
        await self.create_session()
        
        # Đảm bảo không gửi quá nhanh
        current_time = time.time()
        delay = max(0, self.last_message_time + 0.1 - current_time)
        if delay > 0:
            await asyncio.sleep(delay)
        
        try:
            url = f"{self.api_url}/sendDocument"
            
            data = aiohttp.FormData()
            data.add_field("chat_id", str(chat_id))
            
            if isinstance(document, str) and (document.startswith("http://") or document.startswith("https://")):
                # Nếu là URL
                data.add_field("document", document)
            else:
                # Nếu là dữ liệu nhị phân
                if isinstance(document, bytes):
                    document_data = BytesIO(document)
                else:
                    document_data = document
                
                if filename is None:
                    filename = f"document_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                
                data.add_field("document", document_data, filename=filename)
            
            if caption:
                data.add_field("caption", caption)
            
            if parse_mode:
                data.add_field("parse_mode", parse_mode)
            
            data.add_field("disable_notification", str(disable_notification).lower())
            
            if reply_to_message_id:
                data.add_field("reply_to_message_id", str(reply_to_message_id))
            
            async with self.session.post(url, data=data) as response:
                result = await response.json()
                
                if result.get("ok"):
                    self.last_message_time = time.time()
                    return result.get("result")
                else:
                    self.logger.error(f"Lỗi khi gửi tệp: {result.get('description')}")
                    return None
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API sendDocument: {str(e)}", exc_info=True)
            return None
    
    async def edit_message_text(
        self,
        chat_id: Union[int, str],
        message_id: int,
        text: str,
        parse_mode: Optional[str] = "HTML",
        disable_web_page_preview: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Chỉnh sửa tin nhắn văn bản.
        
        Args:
            chat_id: ID chat chứa tin nhắn
            message_id: ID tin nhắn cần chỉnh sửa
            text: Nội dung mới
            parse_mode: Chế độ phân tích ("HTML", "Markdown" hoặc None)
            disable_web_page_preview: Tắt xem trước trang web
            
        Returns:
            Dict kết quả hoặc None nếu có lỗi
        """
        await self.create_session()
        
        try:
            url = f"{self.api_url}/editMessageText"
            payload = {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "disable_web_page_preview": disable_web_page_preview
            }
            
            if parse_mode:
                payload["parse_mode"] = parse_mode
            
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                
                if result.get("ok"):
                    return result.get("result")
                else:
                    self.logger.error(f"Lỗi khi chỉnh sửa tin nhắn: {result.get('description')}")
                    return None
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API editMessageText: {str(e)}", exc_info=True)
            return None
    
    async def delete_message(
        self,
        chat_id: Union[int, str],
        message_id: int
    ) -> bool:
        """
        Xóa tin nhắn.
        
        Args:
            chat_id: ID chat chứa tin nhắn
            message_id: ID tin nhắn cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không
        """
        await self.create_session()
        
        try:
            url = f"{self.api_url}/deleteMessage"
            payload = {
                "chat_id": chat_id,
                "message_id": message_id
            }
            
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                
                if result.get("ok"):
                    return True
                else:
                    self.logger.error(f"Lỗi khi xóa tin nhắn: {result.get('description')}")
                    return False
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API deleteMessage: {str(e)}", exc_info=True)
            return False
    
    async def get_chat(self, chat_id: Union[int, str]) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin chat.
        
        Args:
            chat_id: ID chat cần lấy thông tin
            
        Returns:
            Dict thông tin chat hoặc None nếu có lỗi
        """
        await self.create_session()
        
        try:
            url = f"{self.api_url}/getChat"
            params = {"chat_id": chat_id}
            
            async with self.session.get(url, params=params) as response:
                result = await response.json()
                
                if result.get("ok"):
                    return result.get("result")
                else:
                    self.logger.error(f"Lỗi khi lấy thông tin chat: {result.get('description')}")
                    return None
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API getChat: {str(e)}", exc_info=True)
            return None
    
    async def get_chat_member(
        self,
        chat_id: Union[int, str],
        user_id: int
    ) -> Optional[Dict[str, Any]]:
        """
        Lấy thông tin thành viên chat.
        
        Args:
            chat_id: ID chat
            user_id: ID người dùng
            
        Returns:
            Dict thông tin thành viên hoặc None nếu có lỗi
        """
        await self.create_session()
        
        try:
            url = f"{self.api_url}/getChatMember"
            params = {
                "chat_id": chat_id,
                "user_id": user_id
            }
            
            async with self.session.get(url, params=params) as response:
                result = await response.json()
                
                if result.get("ok"):
                    return result.get("result")
                else:
                    self.logger.error(f"Lỗi khi lấy thông tin thành viên: {result.get('description')}")
                    return None
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API getChatMember: {str(e)}", exc_info=True)
            return None
    
    async def send_chat_action(
        self,
        chat_id: Union[int, str],
        action: str = "typing"
    ) -> bool:
        """
        Gửi trạng thái chat (đang gõ, đang gửi ảnh, v.v.).
        
        Args:
            chat_id: ID chat
            action: Loại hành động ("typing", "upload_photo", "upload_document", v.v.)
            
        Returns:
            True nếu thành công, False nếu không
        """
        await self.create_session()
        
        try:
            url = f"{self.api_url}/sendChatAction"
            payload = {
                "chat_id": chat_id,
                "action": action
            }
            
            async with self.session.post(url, json=payload) as response:
                result = await response.json()
                
                if result.get("ok"):
                    return True
                else:
                    self.logger.error(f"Lỗi khi gửi trạng thái chat: {result.get('description')}")
                    return False
        except Exception as e:
            self.logger.error(f"Lỗi khi gọi API sendChatAction: {str(e)}", exc_info=True)
            return False
    
    async def test_token(self) -> Dict[str, Any]:
        """
        Kiểm tra xem token có hợp lệ không.
        
        Returns:
            Dict chứa kết quả kiểm tra
        """
        bot_info = await self.get_me()
        
        if bot_info:
            return {
                "valid": True,
                "bot_name": bot_info.get("first_name", ""),
                "bot_username": bot_info.get("username", ""),
                "bot_id": bot_info.get("id", 0)
            }
        else:
            return {
                "valid": False,
                "error": "Token không hợp lệ hoặc có lỗi kết nối"
            }

class TelegramChannel:
    """
    Lớp quản lý kênh Telegram.
    Lưu trữ thông tin về một kênh hoặc chat nhận thông báo.
    """
    
    def __init__(
        self,
        chat_id: Union[int, str],
        name: str = "",
        description: str = "",
        is_group: bool = False,
        notification_types: Optional[List[str]] = None
    ):
        """
        Khởi tạo kênh Telegram.
        
        Args:
            chat_id: ID chat
            name: Tên kênh
            description: Mô tả kênh
            is_group: True nếu là nhóm
            notification_types: Danh sách loại thông báo cho phép
        """
        self.chat_id = chat_id
        self.name = name
        self.description = description
        self.is_group = is_group
        self.notification_types = notification_types or ["all"]
        
        # Thời gian thêm
        self.added_time = datetime.now()
        
        # Trạng thái
        self.active = True
        self.last_message_time = None
        self.message_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Chuyển đổi thành dictionary.
        
        Returns:
            Dict chứa thông tin kênh
        """
        return {
            "chat_id": self.chat_id,
            "name": self.name,
            "description": self.description,
            "is_group": self.is_group,
            "notification_types": self.notification_types,
            "added_time": self.added_time.isoformat(),
            "active": self.active,
            "last_message_time": self.last_message_time.isoformat() if self.last_message_time else None,
            "message_count": self.message_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TelegramChannel':
        """
        Tạo đối tượng TelegramChannel từ dictionary.
        
        Args:
            data: Dictionary chứa thông tin kênh
            
        Returns:
            Đối tượng TelegramChannel
        """
        channel = cls(
            chat_id=data["chat_id"],
            name=data.get("name", ""),
            description=data.get("description", ""),
            is_group=data.get("is_group", False),
            notification_types=data.get("notification_types", ["all"])
        )
        
        # Khôi phục các trường khác
        if "added_time" in data:
            channel.added_time = datetime.fromisoformat(data["added_time"])
        
        channel.active = data.get("active", True)
        
        if "last_message_time" in data and data["last_message_time"]:
            channel.last_message_time = datetime.fromisoformat(data["last_message_time"])
        
        channel.message_count = data.get("message_count", 0)
        
        return channel
    
    def can_receive(self, notification_type: str) -> bool:
        """
        Kiểm tra xem kênh có thể nhận loại thông báo hay không.
        
        Args:
            notification_type: Loại thông báo
            
        Returns:
            True nếu có thể nhận, False nếu không
        """
        # Nếu kênh không hoạt động, không thể nhận
        if not self.active:
            return False
        
        # Nếu "all" trong danh sách, có thể nhận tất cả
        if "all" in self.notification_types:
            return True
        
        # Kiểm tra loại thông báo cụ thể
        return notification_type in self.notification_types
    
    def update_stats(self) -> None:
        """
        Cập nhật thống kê sau khi gửi tin nhắn.
        """
        self.last_message_time = datetime.now()
        self.message_count += 1

class TelegramConfiguration:
    """
    Lớp quản lý cấu hình Telegram.
    Lưu trữ các thiết lập của bot và kênh Telegram.
    """
    
    def __init__(
        self,
        token: str = "",
        channels: Optional[List[TelegramChannel]] = None,
        message_template_dir: Optional[Union[str, Path]] = None,
        throttle_limit: int = 30,
        throttle_period: int = 60,
        alert_channels: Optional[Dict[str, List[Union[int, str]]]] = None
    ):
        """
        Khởi tạo cấu hình Telegram.
        
        Args:
            token: Token bot Telegram
            channels: Danh sách kênh
            message_template_dir: Thư mục chứa mẫu tin nhắn
            throttle_limit: Số tin nhắn tối đa trong throttle_period
            throttle_period: Khoảng thời gian giới hạn gửi (giây)
            alert_channels: Dict với key là tên nhóm cảnh báo, value là danh sách ID chat
        """
        self.logger = SystemLogger("telegram_configuration")
        
        # Cài đặt bot
        self.token = token
        self.channels = channels or []
        
        # Cài đặt template
        if message_template_dir is None:
            message_template_dir = Path("config/templates/telegram")
        self.message_template_dir = Path(message_template_dir)
        self.message_template_dir.mkdir(parents=True, exist_ok=True)
        
        # Cài đặt giới hạn
        self.throttle_limit = throttle_limit
        self.throttle_period = throttle_period
        
        # Cài đặt kênh cảnh báo
        self.alert_channels = alert_channels or {
            "system": [],
            "trading": [],
            "security": [],
            "critical": []  # Nhận tất cả cảnh báo Critical
        }
        
        # Danh sách thời gian gửi tin nhắn
        self.sent_timestamps = []
        
        # Template mặc định
        self._create_default_templates()
    
    def _create_default_templates(self) -> None:
        """
        Tạo các mẫu tin nhắn mặc định.
        """
        templates = {
            "alert": """
🚨 <b>{{alert_title}}</b> 🚨

<i>{{alert_time}}</i>

<b>Thông báo:</b> {{alert_message}}

<b>Mức độ:</b> {{alert_level}}
<b>Loại:</b> {{alert_category}}
<b>Nguồn:</b> {{alert_source}}

{{additional_info}}

#{{alert_level}} #{{alert_category}}
            """,
            
            "notification": """
ℹ️ <b>{{notification_title}}</b>

<i>{{notification_time}}</i>

{{notification_message}}

{{notification_content}}

#notification #{{notification_type}}
            """,
            
            "system_status": """
📊 <b>Báo cáo trạng thái hệ thống</b>

<i>{{current_time}}</i>

<b>Tổng thể:</b> {{overall_status}}

<b>CPU:</b> {{cpu_usage}}%
<b>RAM:</b> {{memory_usage}}%
<b>Disk:</b> {{disk_usage}}%

<b>Kết nối mạng:</b> {{network_status}}
<b>Giao dịch:</b> {{trading_status}}

{{additional_info}}

#system #status
            """,
            
            "welcome": """
👋 <b>Chào mừng đến với Automated Trading System!</b>

Chat này đã được thiết lập để nhận thông báo từ hệ thống của chúng tôi.

<b>Cài đặt thông báo:</b>
- Loại thông báo: {{notification_types}}
- Đã đăng ký vào: {{registration_time}}

Để thay đổi cài đặt, vui lòng liên hệ với quản trị viên.

#welcome
            """
        }
        
        # Lưu mẫu vào file
        for name, content in templates.items():
            template_path = self.message_template_dir / f"{name}.txt"
            if not template_path.exists():
                try:
                    with open(template_path, "w", encoding="utf-8") as f:
                        f.write(content.strip())
                    self.logger.debug(f"Đã tạo mẫu tin nhắn {name}")
                except Exception as e:
                    self.logger.error(f"Lỗi khi tạo mẫu tin nhắn {name}: {str(e)}")
    
    def load_template(self, name: str) -> Optional[str]:
        """
        Tải mẫu tin nhắn từ file.
        
        Args:
            name: Tên mẫu
            
        Returns:
            Nội dung mẫu hoặc None nếu không tìm thấy
        """
        template_path = self.message_template_dir / f"{name}.txt"
        
        if not template_path.exists():
            self.logger.warning(f"Không tìm thấy mẫu tin nhắn '{name}'")
            return None
        
        try:
            with open(template_path, "r", encoding="utf-8") as f:
                return f.read()
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mẫu tin nhắn: {str(e)}", exc_info=True)
            return None
    
    def get_template(self, name: str) -> str:
        """
        Lấy mẫu tin nhắn.
        
        Args:
            name: Tên mẫu
            
        Returns:
            Nội dung mẫu (mẫu mặc định nếu không tìm thấy)
        """
        # Thử tải từ file
        template = self.load_template(name)
        
        # Nếu không tìm thấy, sử dụng mẫu mặc định đơn giản
        if template is None:
            if name == "alert":
                template = "🚨 {{alert_title}} 🚨\n\n{{alert_message}}\n\nMức độ: {{alert_level}}\nLoại: {{alert_category}}\nNguồn: {{alert_source}}"
            elif name == "notification":
                template = "ℹ️ {{notification_title}}\n\n{{notification_message}}\n\n{{notification_content}}"
            elif name == "system_status":
                template = "📊 Báo cáo trạng thái hệ thống\n\nTổng thể: {{overall_status}}\nCPU: {{cpu_usage}}%\nRAM: {{memory_usage}}%"
            else:
                template = "{{message}}"
        
        return template
    
    def apply_template(self, name: str, parameters: Dict[str, Any]) -> str:
        """
        Áp dụng tham số vào mẫu.
        
        Args:
            name: Tên mẫu
            parameters: Dict chứa các tham số cần thay thế
            
        Returns:
            Nội dung tin nhắn sau khi đã thay thế tham số
        """
        template = self.get_template(name)
        
        # Thay thế các tham số trong mẫu
        for key, value in parameters.items():
            placeholder = f"{{{{{key}}}}}"
            template = template.replace(placeholder, str(value))
        
        return template
    
    def add_channel(self, channel: TelegramChannel) -> bool:
        """
        Thêm kênh mới.
        
        Args:
            channel: Đối tượng kênh cần thêm
            
        Returns:
            True nếu thêm thành công, False nếu đã tồn tại
        """
        # Kiểm tra xem kênh đã tồn tại chưa
        for existing_channel in self.channels:
            if str(existing_channel.chat_id) == str(channel.chat_id):
                return False
        
        # Thêm kênh mới
        self.channels.append(channel)
        return True
    
    def remove_channel(self, chat_id: Union[int, str]) -> bool:
        """
        Xóa kênh.
        
        Args:
            chat_id: ID chat của kênh cần xóa
            
        Returns:
            True nếu xóa thành công, False nếu không tìm thấy
        """
        for i, channel in enumerate(self.channels):
            if str(channel.chat_id) == str(chat_id):
                self.channels.pop(i)
                return True
        
        return False
    
    def get_channel(self, chat_id: Union[int, str]) -> Optional[TelegramChannel]:
        """
        Lấy thông tin kênh theo ID.
        
        Args:
            chat_id: ID chat cần tìm
            
        Returns:
            Đối tượng TelegramChannel hoặc None nếu không tìm thấy
        """
        for channel in self.channels:
            if str(channel.chat_id) == str(chat_id):
                return channel
        
        return None
    
    def get_channels_for_notification_type(self, notification_type: str) -> List[TelegramChannel]:
        """
        Lấy danh sách kênh cho loại thông báo.
        
        Args:
            notification_type: Loại thông báo
            
        Returns:
            Danh sách kênh có thể nhận loại thông báo
        """
        return [channel for channel in self.channels if channel.can_receive(notification_type)]
    
    def get_channels_for_alert(self, category: str) -> List[TelegramChannel]:
        """
        Lấy danh sách kênh cho loại cảnh báo.
        
        Args:
            category: Loại cảnh báo
            
        Returns:
            Danh sách kênh nhận cảnh báo
        """
        channels = []
        
        # Lấy ID chat từ cài đặt
        chat_ids = set()
        
        # Thêm chat ID theo loại cảnh báo
        if category in self.alert_channels:
            chat_ids.update(self.alert_channels[category])
        
        # Thêm chat ID nhận tất cả cảnh báo Critical
        chat_ids.update(self.alert_channels.get("critical", []))
        
        # Tìm kênh tương ứng
        for chat_id in chat_ids:
            channel = self.get_channel(chat_id)
            if channel:
                channels.append(channel)
        
        return channels
    
    def can_send_message(self) -> bool:
        """
        Kiểm tra xem có thể gửi tin nhắn hay không (giới hạn tốc độ).
        
        Returns:
            True nếu có thể gửi, False nếu không
        """
        now = time.time()
        
        # Xóa các timestamp cũ
        self.sent_timestamps = [ts for ts in self.sent_timestamps if now - ts < self.throttle_period]
        
        # Kiểm tra giới hạn
        return len(self.sent_timestamps) < self.throttle_limit
    
    def record_message_sent(self) -> None:
        """
        Ghi nhận đã gửi một tin nhắn.
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
                "token": self.token,
                "channels": [channel.to_dict() for channel in self.channels],
                "message_template_dir": str(self.message_template_dir),
                "throttle_limit": self.throttle_limit,
                "throttle_period": self.throttle_period,
                "alert_channels": self.alert_channels
            }
            
            # Lưu vào file
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, ensure_ascii=False, indent=4)
            
            self.logger.info(f"Đã lưu cấu hình Telegram vào {file_path}")
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu cấu hình Telegram: {str(e)}", exc_info=True)
            return False
    
    @classmethod
    def load_config(cls, file_path: Union[str, Path]) -> Optional['TelegramConfiguration']:
        """
        Tải cấu hình từ file.
        
        Args:
            file_path: Đường dẫn file
            
        Returns:
            Đối tượng TelegramConfiguration hoặc None nếu không thành công
        """
        logger = SystemLogger("telegram_configuration")
        
        try:
            # Kiểm tra file tồn tại
            if not Path(file_path).exists():
                logger.error(f"Không tìm thấy file cấu hình Telegram: {file_path}")
                return None
            
            # Đọc cấu hình từ file
            with open(file_path, "r", encoding="utf-8") as f:
                config_dict = json.load(f)
            
            # Tạo đối tượng TelegramConfiguration
            config = cls(
                token=config_dict.get("token", ""),
                message_template_dir=config_dict.get("message_template_dir"),
                throttle_limit=config_dict.get("throttle_limit", 30),
                throttle_period=config_dict.get("throttle_period", 60),
                alert_channels=config_dict.get("alert_channels", {})
            )
            
            # Tạo các đối tượng kênh
            for channel_dict in config_dict.get("channels", []):
                channel = TelegramChannel.from_dict(channel_dict)
                config.channels.append(channel)
            
            logger.info(f"Đã tải cấu hình Telegram từ {file_path}")
            return config
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình Telegram: {str(e)}", exc_info=True)
            return None

class TelegramNotifier:
    """
    Lớp thông báo qua Telegram.
    Quản lý việc gửi thông báo và cảnh báo qua Telegram.
    """
    
    def __init__(
        self,
        config: Optional[TelegramConfiguration] = None,
        config_path: Optional[Union[str, Path]] = None,
        token: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo thông báo Telegram.
        
        Args:
            config: Cấu hình Telegram
            config_path: Đường dẫn file cấu hình (nếu không cung cấp config)
            token: Token bot Telegram (nếu không có trong cấu hình)
            logger: Logger tùy chỉnh
        """
        # Logger
        self.logger = logger or SystemLogger("telegram_notifier")
        
        # Tải cấu hình
        if config:
            self.config = config
        elif config_path:
            self.config = TelegramConfiguration.load_config(config_path)
            
            if self.config is None:
                self.logger.warning("Không thể tải cấu hình Telegram, sử dụng cấu hình mặc định")
                self.config = TelegramConfiguration()
        else:
            self.config = TelegramConfiguration()
        
        # Sử dụng token được cung cấp nếu có
        if token:
            self.config.token = token
        
        # Tạo bot
        self.bot = TelegramBot(self.config.token, logger=self.logger)
        
        # Hàng đợi tin nhắn
        self.message_queue = asyncio.Queue()
        
        # Thread gửi tin nhắn
        self.sender_task = None
        self.running = False
        
        # Danh sách hash tin nhắn đã gửi gần đây (để khử trùng)
        self.recent_message_hashes = set()
        
        # Task xử lý cập nhật từ Telegram
        self.update_processor_task = None
        self.last_update_id = 0
        
        # Callbacks
        self.command_handlers = {}
        self.message_handlers = []
    
    async def start(self) -> bool:
        """
        Bắt đầu gửi thông báo.
        
        Returns:
            True nếu bắt đầu thành công, False nếu không
        """
        if self.running:
            self.logger.warning("Thông báo Telegram đã đang chạy")
            return False
        
        # Kiểm tra token
        if not self.config.token:
            self.logger.error("Chưa cấu hình token bot Telegram")
            return False
        
        # Kiểm tra token hợp lệ
        test_result = await self.bot.test_token()
        if not test_result["valid"]:
            self.logger.error(f"Token bot Telegram không hợp lệ: {test_result.get('error')}")
            return False
        
        self.running = True
        
        # Khởi tạo session HTTP
        await self.bot.create_session()
        
        # Khởi động task gửi tin nhắn
        self.sender_task = asyncio.create_task(self._process_message_queue())
        
        # Khởi động task xử lý cập nhật (nếu cần)
        # self.update_processor_task = asyncio.create_task(self._process_updates())
        
        self.logger.info(f"Đã bắt đầu thông báo Telegram với bot @{test_result.get('bot_username')}")
        return True
    
    async def stop(self) -> bool:
        """
        Dừng gửi thông báo.
        
        Returns:
            True nếu dừng thành công, False nếu không
        """
        if not self.running:
            self.logger.warning("Thông báo Telegram không đang chạy")
            return False
        
        self.running = False
        
        # Dừng task gửi tin nhắn
        if self.sender_task:
            try:
                self.sender_task.cancel()
                await asyncio.wait_for(asyncio.shield(self.sender_task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Dừng task xử lý cập nhật
        if self.update_processor_task:
            try:
                self.update_processor_task.cancel()
                await asyncio.wait_for(asyncio.shield(self.update_processor_task), timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        
        # Đóng session HTTP
        await self.bot.close_session()
        
        self.logger.info("Đã dừng thông báo Telegram")
        return True
    
    async def _process_message_queue(self) -> None:
        """
        Vòng lặp xử lý hàng đợi tin nhắn.
        """
        self.logger.info("Bắt đầu vòng lặp xử lý tin nhắn Telegram")
        
        while self.running:
            try:
                # Lấy tin nhắn từ hàng đợi
                message_data = await self.message_queue.get()
                
                # Kiểm tra giới hạn tốc độ
                if not self.config.can_send_message():
                    self.logger.warning("Đã đạt giới hạn gửi tin nhắn, đợi và thử lại sau")
                    # Đặt lại vào hàng đợi
                    await self.message_queue.put(message_data)
                    # Đợi một chút trước khi thử lại
                    await asyncio.sleep(self.config.throttle_period / self.config.throttle_limit)
                    continue
                
                # Gửi tin nhắn
                success = False
                
                if message_data["type"] == "text":
                    # Hiển thị đang gõ...
                    await self.bot.send_chat_action(message_data["chat_id"], "typing")
                    
                    # Gửi tin nhắn văn bản
                    result = await self.bot.send_message(
                        chat_id=message_data["chat_id"],
                        text=message_data["text"],
                        parse_mode=message_data.get("parse_mode", "HTML"),
                        disable_notification=message_data.get("disable_notification", False),
                        disable_web_page_preview=True
                    )
                    success = result is not None
                
                elif message_data["type"] == "photo":
                    # Hiển thị đang gửi ảnh...
                    await self.bot.send_chat_action(message_data["chat_id"], "upload_photo")
                    
                    # Gửi ảnh
                    result = await self.bot.send_photo(
                        chat_id=message_data["chat_id"],
                        photo=message_data["photo"],
                        caption=message_data.get("caption"),
                        parse_mode=message_data.get("parse_mode", "HTML"),
                        disable_notification=message_data.get("disable_notification", False)
                    )
                    success = result is not None
                
                elif message_data["type"] == "document":
                    # Hiển thị đang gửi tệp...
                    await self.bot.send_chat_action(message_data["chat_id"], "upload_document")
                    
                    # Gửi tệp
                    result = await self.bot.send_document(
                        chat_id=message_data["chat_id"],
                        document=message_data["document"],
                        filename=message_data.get("filename"),
                        caption=message_data.get("caption"),
                        parse_mode=message_data.get("parse_mode", "HTML"),
                        disable_notification=message_data.get("disable_notification", False)
                    )
                    success = result is not None
                
                # Đánh dấu đã xử lý
                self.message_queue.task_done()
                
                if success:
                    # Ghi nhận đã gửi thành công
                    self.config.record_message_sent()
                    
                    # Cập nhật thống kê kênh
                    channel = self.config.get_channel(message_data["chat_id"])
                    if channel:
                        channel.update_stats()
                else:
                    self.logger.error(f"Không thể gửi tin nhắn đến chat {message_data['chat_id']}")
                
                # Đợi một chút trước khi gửi tin nhắn tiếp theo (tránh giới hạn Telegram)
                await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp xử lý tin nhắn: {str(e)}", exc_info=True)
                await asyncio.sleep(1)  # Đợi một chút trước khi thử lại
        
        self.logger.info("Kết thúc vòng lặp xử lý tin nhắn Telegram")
    
    async def _process_updates(self) -> None:
        """
        Vòng lặp xử lý các cập nhật từ Telegram.
        """
        self.logger.info("Bắt đầu vòng lặp xử lý cập nhật Telegram")
        
        while self.running:
            try:
                # Lấy cập nhật từ Telegram
                updates = await self.bot.get_updates(
                    offset=self.last_update_id + 1,
                    timeout=30,
                    allowed_updates=["message", "callback_query"]
                )
                
                # Xử lý từng cập nhật
                for update in updates:
                    # Cập nhật last_update_id
                    if update.get("update_id", 0) > self.last_update_id:
                        self.last_update_id = update["update_id"]
                    
                    # Xử lý tin nhắn
                    if "message" in update:
                        await self._handle_message(update["message"])
                    
                    # Xử lý callback query
                    elif "callback_query" in update:
                        await self._handle_callback_query(update["callback_query"])
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Lỗi trong vòng lặp xử lý cập nhật: {str(e)}", exc_info=True)
                await asyncio.sleep(5)  # Đợi một chút trước khi thử lại
        
        self.logger.info("Kết thúc vòng lặp xử lý cập nhật Telegram")
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """
        Xử lý tin nhắn từ người dùng.
        
        Args:
            message: Dữ liệu tin nhắn từ Telegram
        """
        # Kiểm tra xem có phải là lệnh không
        if "text" in message and message["text"].startswith("/"):
            # Lấy tên lệnh và tham số
            command_parts = message["text"].split()
            command = command_parts[0].lower()
            args = command_parts[1:]
            
            # Xử lý lệnh
            handled = False
            
            # Lệnh với @botname
            if "@" in command:
                command_base, bot_name = command.split("@", 1)
                bot_info = await self.bot.get_me()
                
                # Nếu không phải gọi đến bot này, bỏ qua
                if bot_info and bot_name.lower() != bot_info["username"].lower():
                    return
                
                command = command_base
            
            # Tìm handler cho lệnh
            handler = self.command_handlers.get(command)
            if handler:
                try:
                    await handler(message, args)
                    handled = True
                except Exception as e:
                    self.logger.error(f"Lỗi khi xử lý lệnh {command}: {str(e)}", exc_info=True)
            
            # Xử lý các lệnh mặc định nếu chưa được xử lý
            if not handled:
                if command == "/start":
                    await self._handle_start_command(message)
                elif command == "/help":
                    await self._handle_help_command(message)
                elif command == "/status":
                    await self._handle_status_command(message)
        
        # Xử lý tin nhắn thông thường
        else:
            # Gọi tất cả các message handler
            for handler in self.message_handlers:
                try:
                    await handler(message)
                except Exception as e:
                    self.logger.error(f"Lỗi khi xử lý tin nhắn: {str(e)}", exc_info=True)
    
    async def _handle_callback_query(self, callback_query: Dict[str, Any]) -> None:
        """
        Xử lý callback query từ nút nhấn inline.
        
        Args:
            callback_query: Dữ liệu callback query từ Telegram
        """
        # Phần này sẽ được triển khai sau khi cần xử lý inline keyboard
        pass
    
    async def _handle_start_command(self, message: Dict[str, Any]) -> None:
        """
        Xử lý lệnh /start.
        
        Args:
            message: Dữ liệu tin nhắn từ Telegram
        """
        chat_id = message["chat"]["id"]
        
        # Kiểm tra xem chat đã đăng ký chưa
        channel = self.config.get_channel(chat_id)
        
        if channel:
            # Chat đã đăng ký
            welcome_text = (
                f"Xin chào! Chat này đã được đăng ký nhận thông báo từ Automated Trading System.\n\n"
                f"Loại thông báo: {', '.join(channel.notification_types)}\n\n"
                f"Sử dụng /help để xem các lệnh khả dụng."
            )
        else:
            # Chat chưa đăng ký
            name = ""
            is_group = False
            
            if message["chat"]["type"] == "private":
                name = message["chat"].get("first_name", "") + " " + message["chat"].get("last_name", "")
                name = name.strip()
            else:
                name = message["chat"].get("title", "")
                is_group = True
            
            # Tạo kênh mới
            new_channel = TelegramChannel(
                chat_id=chat_id,
                name=name,
                is_group=is_group,
                notification_types=["all"]
            )
            
            # Thêm vào danh sách
            self.config.add_channel(new_channel)
            
            # Áp dụng mẫu welcome
            welcome_text = self.config.apply_template("welcome", {
                "notification_types": "Tất cả",
                "registration_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            })
        
        # Gửi tin nhắn chào mừng
        await self.bot.send_message(chat_id, welcome_text)
    
    async def _handle_help_command(self, message: Dict[str, Any]) -> None:
        """
        Xử lý lệnh /help.
        
        Args:
            message: Dữ liệu tin nhắn từ Telegram
        """
        chat_id = message["chat"]["id"]
        
        help_text = (
            "<b>Các lệnh khả dụng:</b>\n\n"
            "/start - Khởi động bot và đăng ký nhận thông báo\n"
            "/help - Hiển thị trợ giúp này\n"
            "/status - Hiển thị trạng thái hệ thống hiện tại\n\n"
            "Bot này sẽ gửi thông báo tự động khi có sự kiện quan trọng từ hệ thống giao dịch tự động."
        )
        
        await self.bot.send_message(chat_id, help_text)
    
    async def _handle_status_command(self, message: Dict[str, Any]) -> None:
        """
        Xử lý lệnh /status.
        
        Args:
            message: Dữ liệu tin nhắn từ Telegram
        """
        chat_id = message["chat"]["id"]
        
        await self.bot.send_message(
            chat_id,
            "Đang lấy trạng thái hệ thống, vui lòng đợi..."
        )
        
        try:
            # Tạo nội dung trạng thái giả định
            # Trong ứng dụng thực tế, sẽ lấy thông tin từ hệ thống thực
            import psutil
            
            cpu_usage = psutil.cpu_percent(interval=1)
            memory_usage = psutil.virtual_memory().percent
            disk_usage = psutil.disk_usage(Path.cwd()).percent
            
            # Xác định trạng thái tổng thể
            if max(cpu_usage, memory_usage, disk_usage) > 90:
                overall_status = "⚠️ Cảnh báo"
            elif max(cpu_usage, memory_usage, disk_usage) > 70:
                overall_status = "🟡 Chú ý"
            else:
                overall_status = "✅ Bình thường"
            
            # Áp dụng mẫu system_status
            status_text = self.config.apply_template("system_status", {
                "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "overall_status": overall_status,
                "cpu_usage": cpu_usage,
                "memory_usage": memory_usage,
                "disk_usage": disk_usage,
                "network_status": "✅ Đang hoạt động",
                "trading_status": "✅ Đang hoạt động",
                "additional_info": ""
            })
            
            await self.bot.send_message(chat_id, status_text)
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xử lý lệnh status: {str(e)}", exc_info=True)
            await self.bot.send_message(
                chat_id,
                "❌ Lỗi khi lấy trạng thái hệ thống. Vui lòng thử lại sau."
            )
    
    def register_command_handler(self, command: str, handler: Callable) -> None:
        """
        Đăng ký handler cho lệnh.
        
        Args:
            command: Tên lệnh (bắt đầu bằng /)
            handler: Hàm xử lý nhận (message, args)
        """
        # Đảm bảo lệnh bắt đầu bằng /
        if not command.startswith("/"):
            command = f"/{command}"
        
        self.command_handlers[command] = handler
        self.logger.debug(f"Đã đăng ký handler cho lệnh {command}")
    
    def register_message_handler(self, handler: Callable) -> None:
        """
        Đăng ký handler cho tin nhắn thông thường.
        
        Args:
            handler: Hàm xử lý nhận message
        """
        self.message_handlers.append(handler)
        self.logger.debug(f"Đã đăng ký message handler")
    
    def _calculate_message_hash(self, text: str) -> str:
        """
        Tính toán hash cho tin nhắn để khử trùng.
        
        Args:
            text: Nội dung tin nhắn
            
        Returns:
            Hash tin nhắn
        """
        import hashlib
        
        # Lấy 100 ký tự đầu của tin nhắn để tính hash
        content = text[:100]
        
        # Băm nội dung
        return hashlib.md5(content.encode()).hexdigest()
    
    async def _remove_message_hash(self, message_hash: str) -> None:
        """
        Xóa hash tin nhắn sau khoảng thời gian.
        
        Args:
            message_hash: Hash tin nhắn cần xóa
        """
        await asyncio.sleep(3600)  # 1 giờ
        self.recent_message_hashes.discard(message_hash)
    
    def _get_alert_header_emoji(self, level: AlertLevel) -> str:
        """
        Lấy emoji header cho cấp độ cảnh báo.
        
        Args:
            level: Cấp độ cảnh báo
            
        Returns:
            Emoji thích hợp
        """
        emojis = {
            AlertLevel.INFO: "ℹ️",
            AlertLevel.NOTICE: "📝",
            AlertLevel.WARNING: "⚠️",
            AlertLevel.ALERT: "🚨",
            AlertLevel.CRITICAL: "🔥"
        }
        
        return emojis.get(level, "ℹ️")
    
    async def send_alert(self, alert: Alert) -> bool:
        """
        Gửi cảnh báo qua Telegram.
        
        Args:
            alert: Đối tượng cảnh báo
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        if not self.running or not self.config.token:
            self.logger.error("Thông báo Telegram chưa được khởi động hoặc chưa cấu hình token")
            return False
        
        # Tạo tiêu đề
        header_emoji = self._get_alert_header_emoji(alert.level)
        alert_title = f"{header_emoji} {alert.category.name.upper()}: {alert.level.name}"
        
        # Tạo tham số
        parameters = {
            "alert_title": alert_title,
            "alert_message": alert.message,
            "alert_time": alert.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            "alert_level": alert.level.name,
            "alert_category": alert.category.name,
            "alert_source": alert.source,
            "additional_info": ""
        }
        
        # Thêm thông tin bổ sung nếu có
        if alert.data:
            additional_info = "<b>Thông tin thêm:</b>\n"
            for key, value in alert.data.items():
                additional_info += f"• <b>{key}:</b> {value}\n"
            parameters["additional_info"] = additional_info
        
        # Áp dụng mẫu
        text = self.config.apply_template("alert", parameters)
        
        # Tính hash để khử trùng
        message_hash = self._calculate_message_hash(text)
        
        # Kiểm tra trùng lặp
        if message_hash in self.recent_message_hashes:
            self.logger.debug("Bỏ qua cảnh báo trùng lặp")
            return False
        
        # Thêm hash vào danh sách gần đây
        self.recent_message_hashes.add(message_hash)
        
        # Lên lịch xóa hash sau khoảng thời gian
        asyncio.create_task(self._remove_message_hash(message_hash))
        
        # Lấy danh sách kênh
        channels = self.config.get_channels_for_alert(alert.category.name.lower())
        
        # Thêm kênh nhận tất cả cảnh báo critical
        if alert.level == AlertLevel.CRITICAL:
            critical_channels = self.config.get_channels_for_alert("critical")
            channels.extend(critical_channels)
        
        # Loại bỏ trùng lặp
        channels = list({channel.chat_id: channel for channel in channels}.values())
        
        if not channels:
            self.logger.warning(f"Không có kênh nào để gửi cảnh báo {alert.alert_id}")
            return False
        
        # Tạo thông báo im lặng nếu cấp độ thấp
        disable_notification = alert.level in [AlertLevel.INFO, AlertLevel.NOTICE]
        
        # Thêm vào hàng đợi cho từng kênh
        for channel in channels:
            await self.message_queue.put({
                "type": "text",
                "chat_id": channel.chat_id,
                "text": text,
                "parse_mode": "HTML",
                "disable_notification": disable_notification
            })
        
        self.logger.info(f"Đã thêm cảnh báo vào hàng đợi Telegram cho {len(channels)} kênh")
        return True
    
    async def send_notification(
        self,
        title: str,
        message: str,
        notification_type: str = "general",
        content: str = "",
        photo: Optional[Union[str, bytes, BytesIO]] = None,
        document: Optional[Union[str, bytes, BytesIO]] = None,
        filename: Optional[str] = None,
        channels: Optional[List[Union[int, str]]] = None
    ) -> bool:
        """
        Gửi thông báo qua Telegram.
        
        Args:
            title: Tiêu đề thông báo
            message: Nội dung thông báo
            notification_type: Loại thông báo
            content: Nội dung HTML bổ sung
            photo: Ảnh đính kèm
            document: Tệp đính kèm
            filename: Tên tệp
            channels: Danh sách ID chat (None để gửi cho tất cả)
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        if not self.running or not self.config.token:
            self.logger.error("Thông báo Telegram chưa được khởi động hoặc chưa cấu hình token")
            return False
        
        # Tạo tham số
        parameters = {
            "notification_title": title,
            "notification_message": message,
            "notification_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "notification_type": notification_type,
            "notification_content": content
        }
        
        # Áp dụng mẫu
        text = self.config.apply_template("notification", parameters)
        
        # Tính hash để khử trùng
        message_hash = self._calculate_message_hash(text)
        
        # Kiểm tra trùng lặp
        if message_hash in self.recent_message_hashes:
            self.logger.debug("Bỏ qua thông báo trùng lặp")
            return False
        
        # Thêm hash vào danh sách gần đây
        self.recent_message_hashes.add(message_hash)
        
        # Lên lịch xóa hash sau khoảng thời gian
        asyncio.create_task(self._remove_message_hash(message_hash))
        
        # Xác định danh sách kênh
        target_channels = []
        
        if channels:
            # Sử dụng danh sách cung cấp
            for chat_id in channels:
                channel = self.config.get_channel(chat_id)
                if channel and channel.can_receive(notification_type):
                    target_channels.append(channel)
        else:
            # Lấy tất cả kênh có thể nhận loại thông báo này
            target_channels = self.config.get_channels_for_notification_type(notification_type)
        
        if not target_channels:
            self.logger.warning(f"Không có kênh nào để gửi thông báo loại {notification_type}")
            return False
        
        # Quyết định loại tin nhắn cần gửi
        if photo:
            # Gửi ảnh với chú thích
            for channel in target_channels:
                await self.message_queue.put({
                    "type": "photo",
                    "chat_id": channel.chat_id,
                    "photo": photo,
                    "caption": text if len(text) <= 1024 else text[:1021] + "...",
                    "parse_mode": "HTML"
                })
        elif document:
            # Gửi tệp với chú thích
            for channel in target_channels:
                await self.message_queue.put({
                    "type": "document",
                    "chat_id": channel.chat_id,
                    "document": document,
                    "filename": filename,
                    "caption": text if len(text) <= 1024 else text[:1021] + "...",
                    "parse_mode": "HTML"
                })
        else:
            # Gửi tin nhắn văn bản
            for channel in target_channels:
                await self.message_queue.put({
                    "type": "text",
                    "chat_id": channel.chat_id,
                    "text": text,
                    "parse_mode": "HTML"
                })
        
        self.logger.info(f"Đã thêm thông báo vào hàng đợi Telegram cho {len(target_channels)} kênh")
        return True
    
    async def send_system_status(
        self,
        overall_status: str,
        cpu_usage: float,
        memory_usage: float,
        disk_usage: float,
        network_status: str = "Đang hoạt động",
        trading_status: str = "Đang hoạt động",
        additional_info: str = ""
    ) -> bool:
        """
        Gửi thông báo trạng thái hệ thống.
        
        Args:
            overall_status: Trạng thái tổng thể
            cpu_usage: Phần trăm sử dụng CPU
            memory_usage: Phần trăm sử dụng RAM
            disk_usage: Phần trăm sử dụng ổ đĩa
            network_status: Trạng thái mạng
            trading_status: Trạng thái giao dịch
            additional_info: Thông tin bổ sung
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        if not self.running or not self.config.token:
            self.logger.error("Thông báo Telegram chưa được khởi động hoặc chưa cấu hình token")
            return False
        
        # Thêm emoji cho trạng thái
        if "cảnh báo" in overall_status.lower() or "warning" in overall_status.lower():
            overall_status = f"⚠️ {overall_status}"
        elif "bình thường" in overall_status.lower() or "normal" in overall_status.lower():
            overall_status = f"✅ {overall_status}"
        elif "lỗi" in overall_status.lower() or "error" in overall_status.lower():
            overall_status = f"❌ {overall_status}"
        
        # Tạo tham số
        parameters = {
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "overall_status": overall_status,
            "cpu_usage": f"{cpu_usage:.1f}",
            "memory_usage": f"{memory_usage:.1f}",
            "disk_usage": f"{disk_usage:.1f}",
            "network_status": network_status,
            "trading_status": trading_status,
            "additional_info": additional_info
        }
        
        # Áp dụng mẫu
        text = self.config.apply_template("system_status", parameters)
        
        # Lấy tất cả kênh nhận thông báo hệ thống
        system_channels = self.config.get_channels_for_notification_type("system")
        
        if not system_channels:
            self.logger.warning("Không có kênh nào để gửi trạng thái hệ thống")
            return False
        
        # Thêm vào hàng đợi
        for channel in system_channels:
            await self.message_queue.put({
                "type": "text",
                "chat_id": channel.chat_id,
                "text": text,
                "parse_mode": "HTML"
            })
        
        self.logger.info(f"Đã thêm trạng thái hệ thống vào hàng đợi Telegram cho {len(system_channels)} kênh")
        return True
    
    async def send_custom_message(
        self,
        chat_id: Union[int, str],
        text: str,
        parse_mode: Optional[str] = "HTML",
        disable_notification: bool = False
    ) -> bool:
        """
        Gửi tin nhắn tùy chỉnh.
        
        Args:
            chat_id: ID chat nhận tin nhắn
            text: Nội dung tin nhắn
            parse_mode: Chế độ phân tích ("HTML", "Markdown" hoặc None)
            disable_notification: Tắt thông báo
            
        Returns:
            True nếu đã thêm vào hàng đợi, False nếu không
        """
        if not self.running or not self.config.token:
            self.logger.error("Thông báo Telegram chưa được khởi động hoặc chưa cấu hình token")
            return False
        
        # Thêm vào hàng đợi
        await self.message_queue.put({
            "type": "text",
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_notification": disable_notification
        })
        
        self.logger.debug(f"Đã thêm tin nhắn tùy chỉnh vào hàng đợi Telegram cho chat {chat_id}")
        return True

# Singleton instance
_telegram_notifier_instance = None

def get_telegram_notifier() -> TelegramNotifier:
    """
    Lấy instance singleton của TelegramNotifier.
    
    Returns:
        Instance TelegramNotifier
    """
    global _telegram_notifier_instance
    if _telegram_notifier_instance is None:
        _telegram_notifier_instance = TelegramNotifier()
    return _telegram_notifier_instance