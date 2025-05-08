#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Lệnh CLI cho quản lý Dashboard.
File này cung cấp các lệnh để quản lý và tương tác với dashboard của
hệ thống giao dịch tự động qua giao diện dòng lệnh (CLI).
"""

import os
import sys
import time
import signal
import subprocess
import json
import socket
import psutil
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import click
from tabulate import tabulate
import requests

# Thêm đường dẫn gốc vào sys.path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(ROOT_DIR))

# Import các module từ hệ thống
from config.system_config import get_system_config, SystemConfig
from config.logging_config import get_logger
from config.env import get_env, set_env

# Logger
logger = get_logger("dashboard_commands")

# Cấu hình hệ thống
system_config = get_system_config()

# Đường dẫn đến file dashboard
DASHBOARD_APP_PATH = ROOT_DIR / "streamlit_dashboard" / "app.py"
# Tên file pid mặc định
DEFAULT_PID_FILE = ROOT_DIR / "logs" / "dashboard.pid"
# File cấu hình dashboard
DASHBOARD_CONFIG_FILE = ROOT_DIR / "config" / "dashboard_config.json"


class DashboardManager:
    """
    Lớp quản lý Dashboard Streamlit.
    Cung cấp các phương thức để khởi động, dừng và quản lý dashboard.
    """

    def __init__(
        self, 
        app_path: Union[str, Path] = DASHBOARD_APP_PATH,
        pid_file: Union[str, Path] = DEFAULT_PID_FILE,
        config_file: Union[str, Path] = DASHBOARD_CONFIG_FILE
    ):
        """
        Khởi tạo DashboardManager.
        
        Args:
            app_path: Đường dẫn đến file ứng dụng Streamlit
            pid_file: Đường dẫn đến file lưu PID
            config_file: Đường dẫn đến file cấu hình dashboard
        """
        self.app_path = Path(app_path)
        self.pid_file = Path(pid_file)
        self.config_file = Path(config_file)
        
        # Đảm bảo thư mục logs tồn tại
        self.pid_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Tạo file cấu hình nếu chưa tồn tại
        self._ensure_config_file()
    
    def _ensure_config_file(self) -> None:
        """Đảm bảo file cấu hình tồn tại với các giá trị mặc định."""
        if not self.config_file.exists():
            # Tạo thư mục chứa file nếu cần
            self.config_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Cấu hình mặc định
            default_config = {
                "port": 8501,
                "host": "localhost",
                "theme": "light",
                "enable_cache": True,
                "debug": False,
                "allow_remote_access": False,
                "enable_metrics": True,
                "memory_limit": 1000,  # MB
                "startup_timeout": 30,  # seconds
                "title": "Automated Trading System Dashboard",
                "language": "vi",
                "auto_reload_on_file_change": True
            }
            
            # Lưu cấu hình mặc định
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(default_config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Đã tạo file cấu hình dashboard tại: {self.config_file}")
    
    def load_config(self) -> Dict[str, Any]:
        """
        Tải cấu hình dashboard.
        
        Returns:
            Dict chứa cấu hình dashboard
        """
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Lỗi khi tải cấu hình dashboard: {e}")
            # Trả về cấu hình mặc định
            return {"port": 8501, "host": "localhost", "theme": "light"}
    
    def save_config(self, config: Dict[str, Any]) -> bool:
        """
        Lưu cấu hình dashboard.
        
        Args:
            config: Dict chứa cấu hình cần lưu
            
        Returns:
            True nếu lưu thành công, False nếu lỗi
        """
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4, ensure_ascii=False)
            
            logger.info(f"Đã lưu cấu hình dashboard vào: {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Lỗi khi lưu cấu hình dashboard: {e}")
            return False
    
    def is_port_in_use(self, port: int) -> bool:
        """
        Kiểm tra xem một cổng có đang được sử dụng hay không.
        
        Args:
            port: Số cổng cần kiểm tra
            
        Returns:
            True nếu cổng đang được sử dụng, False nếu không
        """
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    
    def find_available_port(self, start_port: int = 8501, max_tries: int = 10) -> int:
        """
        Tìm một cổng khả dụng bắt đầu từ start_port.
        
        Args:
            start_port: Cổng bắt đầu kiểm tra
            max_tries: Số lần thử tối đa
            
        Returns:
            Cổng khả dụng đầu tiên tìm thấy, hoặc start_port nếu không tìm thấy
        """
        current_port = start_port
        for _ in range(max_tries):
            if not self.is_port_in_use(current_port):
                return current_port
            current_port += 1
        
        logger.warning(f"Không tìm thấy cổng khả dụng sau {max_tries} lần thử")
        return start_port  # Trả về cổng ban đầu mặc dù có thể đang bị sử dụng
    
    def save_pid(self, pid: int) -> None:
        """
        Lưu PID vào file.
        
        Args:
            pid: Process ID cần lưu
        """
        with open(self.pid_file, 'w') as f:
            f.write(str(pid))
        
        logger.debug(f"Đã lưu PID {pid} vào file: {self.pid_file}")
    
    def read_pid(self) -> Optional[int]:
        """
        Đọc PID từ file.
        
        Returns:
            PID nếu tồn tại, None nếu không
        """
        if not self.pid_file.exists():
            return None
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            return pid
        except Exception as e:
            logger.error(f"Lỗi khi đọc PID từ file: {e}")
            return None
    
    def is_dashboard_running(self) -> bool:
        """
        Kiểm tra xem dashboard có đang chạy hay không.
        
        Returns:
            True nếu dashboard đang chạy, False nếu không
        """
        pid = self.read_pid()
        if pid is None:
            return False
        
        try:
            # Kiểm tra xem process có tồn tại không
            process = psutil.Process(pid)
            
            # Kiểm tra xem có phải là process Streamlit không
            if "streamlit" in process.name().lower() or "streamlit" in " ".join(process.cmdline()).lower():
                return True
            
            # Nếu không phải là Streamlit, có thể PID file không chính xác
            logger.warning(f"Process với PID {pid} không phải là Streamlit")
            return False
        except psutil.NoSuchProcess:
            logger.debug(f"Process với PID {pid} không tồn tại")
            return False
    
    def get_dashboard_info(self) -> Dict[str, Any]:
        """
        Lấy thông tin về dashboard đang chạy.
        
        Returns:
            Dict chứa thông tin về dashboard
        """
        info = {
            "running": False,
            "pid": None,
            "port": None,
            "url": None,
            "uptime": None,
            "memory_usage": None
        }
        
        pid = self.read_pid()
        if pid is None:
            return info
        
        try:
            # Lấy thông tin process
            process = psutil.Process(pid)
            
            # Kiểm tra xem có phải là process Streamlit không
            cmdline = " ".join(process.cmdline())
            if "streamlit" not in process.name().lower() and "streamlit" not in cmdline.lower():
                logger.warning(f"Process với PID {pid} không phải là Streamlit")
                return info
            
            # Lấy cổng từ command line
            port = 8501  # Mặc định
            for arg in process.cmdline():
                if "--server.port" in arg:
                    port = int(arg.split("=")[1])
                elif arg.isdigit() and process.cmdline()[process.cmdline().index(arg) - 1] == "--server.port":
                    port = int(arg)
            
            # Cập nhật thông tin
            info["running"] = True
            info["pid"] = pid
            info["port"] = port
            info["url"] = f"http://localhost:{port}"
            info["uptime"] = time.time() - process.create_time()
            info["memory_usage"] = process.memory_info().rss / (1024 * 1024)  # MB
            
            return info
        except psutil.NoSuchProcess:
            logger.debug(f"Process với PID {pid} không tồn tại")
            return info
    
    def start_dashboard(
        self, 
        port: Optional[int] = None, 
        host: str = "localhost",
        theme: str = "light",
        debug: bool = False
    ) -> Tuple[bool, str]:
        """
        Khởi động dashboard Streamlit.
        
        Args:
            port: Cổng để chạy dashboard
            host: Địa chỉ host
            theme: Chủ đề giao diện (light/dark)
            debug: Bật chế độ debug
            
        Returns:
            Tuple (success, message)
        """
        # Kiểm tra xem dashboard đã chạy chưa
        if self.is_dashboard_running():
            info = self.get_dashboard_info()
            return False, f"Dashboard đã đang chạy tại: http://{host}:{info['port']}"
        
        # Tải cấu hình
        config = self.load_config()
        
        # Sử dụng các tham số được cung cấp hoặc mặc định từ cấu hình
        if port is None:
            port = config.get("port", 8501)
        
        # Kiểm tra xem cổng có sẵn không
        if self.is_port_in_use(port):
            new_port = self.find_available_port(port)
            logger.warning(f"Cổng {port} đã được sử dụng, chuyển sang cổng {new_port}")
            port = new_port
        
        # Xây dựng command để chạy Streamlit
        cmd = [
            "streamlit",
            "run",
            str(self.app_path),
            "--server.port", str(port),
            "--server.address", host,
            "--theme.base", theme
        ]
        
        if debug:
            cmd.append("--logger.level=debug")
        
        # Cho phép truy cập từ xa
        if config.get("allow_remote_access", False):
            cmd.append("--server.headless")
            cmd.append("--server.enableCORS=false")
            cmd.append("--server.enableXsrfProtection=false")
        
        # Thiết lập tiêu đề
        if "title" in config:
            cmd.extend(["--server.headless", "--client.toolbarMode=minimal"])
            os.environ["STREAMLIT_PAGE_TITLE"] = config["title"]
        
        # Thiết lập ngôn ngữ
        if "language" in config:
            os.environ["STREAMLIT_LANGUAGE"] = config["language"]
        
        # Bật/tắt cache
        if not config.get("enable_cache", True):
            cmd.append("--server.enableCaching=false")
        
        # Bật/tắt tự động tải lại
        if not config.get("auto_reload_on_file_change", True):
            cmd.append("--server.runOnSave=false")
        
        try:
            # Khởi động process
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8"
            )
            
            # Lưu PID
            self.save_pid(process.pid)
            
            # Đợi khởi động Streamlit
            timeout = config.get("startup_timeout", 30)
            start_time = time.time()
            
            while time.time() - start_time < timeout:
                if self.is_port_in_use(port):
                    # Đợi thêm 2 giây để Streamlit khởi động hoàn tất
                    time.sleep(2)
                    url = f"http://{host}:{port}"
                    logger.info(f"Dashboard đã khởi động thành công tại: {url}")
                    return True, f"Dashboard đã khởi động thành công tại: {url}"
                
                # Kiểm tra xem process còn sống không
                if process.poll() is not None:
                    stderr = process.stderr.read()
                    return False, f"Dashboard khởi động thất bại: {stderr}"
                
                time.sleep(0.5)
            
            # Nếu timeout
            return False, f"Dashboard khởi động timeout sau {timeout} giây"
        
        except Exception as e:
            logger.error(f"Lỗi khi khởi động dashboard: {e}")
            return False, f"Lỗi khi khởi động dashboard: {str(e)}"
    
    def stop_dashboard(self, force: bool = False) -> Tuple[bool, str]:
        """
        Dừng dashboard đang chạy.
        
        Args:
            force: Buộc dừng nếu True
            
        Returns:
            Tuple (success, message)
        """
        pid = self.read_pid()
        if pid is None:
            return False, "Không tìm thấy file PID, dashboard có thể không chạy"
        
        try:
            # Lấy process
            process = psutil.Process(pid)
            
            # Kiểm tra xem có phải là process Streamlit không
            if "streamlit" not in process.name().lower() and "streamlit" not in " ".join(process.cmdline()).lower():
                logger.warning(f"Process với PID {pid} không phải là Streamlit")
                return False, f"Process với PID {pid} không phải là Streamlit"
            
            # Dừng process
            if force:
                process.kill()
            else:
                process.terminate()
            
            # Đợi process kết thúc
            timeout = 10  # Số giây tối đa để đợi
            for _ in range(timeout * 2):
                if not psutil.pid_exists(pid):
                    # Xóa file PID
                    if self.pid_file.exists():
                        self.pid_file.unlink()
                    
                    logger.info(f"Đã dừng dashboard với PID {pid}")
                    return True, f"Đã dừng dashboard với PID {pid}"
                
                time.sleep(0.5)
            
            # Nếu vẫn chưa dừng, buộc dừng
            if psutil.pid_exists(pid):
                process.kill()
                logger.warning(f"Buộc dừng dashboard với PID {pid}")
                
                # Xóa file PID
                if self.pid_file.exists():
                    self.pid_file.unlink()
                
                return True, f"Đã buộc dừng dashboard với PID {pid}"
            
            return True, f"Đã dừng dashboard với PID {pid}"
        
        except psutil.NoSuchProcess:
            # Xóa file PID nếu tồn tại
            if self.pid_file.exists():
                self.pid_file.unlink()
            
            logger.info(f"Process với PID {pid} không tồn tại")
            return True, f"Process với PID {pid} không tồn tại, đã xóa file PID"
        
        except Exception as e:
            logger.error(f"Lỗi khi dừng dashboard: {e}")
            return False, f"Lỗi khi dừng dashboard: {str(e)}"
    
    def restart_dashboard(self) -> Tuple[bool, str]:
        """
        Khởi động lại dashboard.
        
        Returns:
            Tuple (success, message)
        """
        # Lấy thông tin dashboard hiện tại
        info = self.get_dashboard_info()
        
        # Dừng dashboard hiện tại
        success, message = self.stop_dashboard()
        if not success:
            return False, f"Không thể dừng dashboard hiện tại: {message}"
        
        # Đợi một chút để đảm bảo cổng được giải phóng
        time.sleep(2)
        
        # Khởi động lại với cùng cấu hình
        port = info.get("port") if info.get("running") else None
        return self.start_dashboard(port=port)
    
    def update_config_param(self, param: str, value: Any) -> Tuple[bool, str]:
        """
        Cập nhật một tham số cấu hình.
        
        Args:
            param: Tên tham số
            value: Giá trị mới
            
        Returns:
            Tuple (success, message)
        """
        try:
            # Tải cấu hình hiện tại
            config = self.load_config()
            
            # Cập nhật tham số
            config[param] = value
            
            # Lưu cấu hình
            if self.save_config(config):
                return True, f"Đã cập nhật {param} = {value}"
            else:
                return False, f"Không thể lưu cấu hình"
        
        except Exception as e:
            logger.error(f"Lỗi khi cập nhật cấu hình: {e}")
            return False, f"Lỗi khi cập nhật cấu hình: {str(e)}"


@click.group(help="Lệnh quản lý Dashboard")
def dashboard():
    """Nhóm lệnh quản lý Dashboard."""
    pass


@dashboard.command(help="Khởi động dashboard")
@click.option("--port", "-p", type=int, help="Cổng để chạy dashboard")
@click.option("--host", "-h", default="localhost", help="Địa chỉ host")
@click.option("--theme", "-t", type=click.Choice(["light", "dark"]), default="light", help="Chủ đề giao diện")
@click.option("--debug/--no-debug", default=False, help="Bật/tắt chế độ debug")
def start(port: Optional[int], host: str, theme: str, debug: bool):
    """
    Khởi động dashboard Streamlit.
    
    Args:
        port: Cổng để chạy dashboard
        host: Địa chỉ host
        theme: Chủ đề giao diện
        debug: Bật chế độ debug
    """
    manager = DashboardManager()
    success, message = manager.start_dashboard(port, host, theme, debug)
    
    if success:
        click.secho(message, fg="green")
    else:
        click.secho(message, fg="red")


@dashboard.command(help="Dừng dashboard đang chạy")
@click.option("--force", "-f", is_flag=True, help="Buộc dừng dashboard")
def stop(force: bool):
    """
    Dừng dashboard đang chạy.
    
    Args:
        force: Buộc dừng nếu True
    """
    manager = DashboardManager()
    success, message = manager.stop_dashboard(force)
    
    if success:
        click.secho(message, fg="green")
    else:
        click.secho(message, fg="red")


@dashboard.command(help="Khởi động lại dashboard")
def restart():
    """Khởi động lại dashboard."""
    manager = DashboardManager()
    success, message = manager.restart_dashboard()
    
    if success:
        click.secho(message, fg="green")
    else:
        click.secho(message, fg="red")


@dashboard.command(help="Kiểm tra trạng thái dashboard")
@click.option("--json", "json_output", is_flag=True, help="Xuất kết quả dưới dạng JSON")
def status(json_output: bool):
    """
    Kiểm tra trạng thái dashboard.
    
    Args:
        json_output: Xuất kết quả dưới dạng JSON nếu True
    """
    manager = DashboardManager()
    info = manager.get_dashboard_info()
    
    if json_output:
        click.echo(json.dumps(info, indent=2))
        return
    
    if info["running"]:
        click.secho("✅ Dashboard đang chạy", fg="green")
        
        # Định dạng thời gian chạy
        uptime_seconds = info["uptime"]
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        uptime_str = ""
        if days > 0:
            uptime_str += f"{int(days)} ngày "
        if hours > 0 or days > 0:
            uptime_str += f"{int(hours)} giờ "
        if minutes > 0 or hours > 0 or days > 0:
            uptime_str += f"{int(minutes)} phút "
        uptime_str += f"{int(seconds)} giây"
        
        # Tạo bảng thông tin
        table_data = [
            ["PID", info["pid"]],
            ["URL", info["url"]],
            ["Thời gian chạy", uptime_str],
            ["Bộ nhớ sử dụng", f"{info['memory_usage']:.2f} MB"]
        ]
        
        click.echo(tabulate(table_data, tablefmt="simple"))
    else:
        click.secho("❌ Dashboard không chạy", fg="red")


@dashboard.command(help="Hiển thị cấu hình dashboard")
@click.option("--json", "json_output", is_flag=True, help="Xuất kết quả dưới dạng JSON")
def config(json_output: bool):
    """
    Hiển thị cấu hình dashboard.
    
    Args:
        json_output: Xuất kết quả dưới dạng JSON nếu True
    """
    manager = DashboardManager()
    config_data = manager.load_config()
    
    if json_output:
        click.echo(json.dumps(config_data, indent=2, ensure_ascii=False))
        return
    
    click.secho("Cấu hình Dashboard:", fg="blue")
    
    # Nhóm các cấu hình
    network_config = {k: v for k, v in config_data.items() if k in ["port", "host", "allow_remote_access"]}
    ui_config = {k: v for k, v in config_data.items() if k in ["theme", "title", "language"]}
    performance_config = {k: v for k, v in config_data.items() if k in ["enable_cache", "memory_limit", "startup_timeout"]}
    other_config = {k: v for k, v in config_data.items() if k not in list(network_config.keys()) + list(ui_config.keys()) + list(performance_config.keys())}
    
    # Hiển thị theo nhóm
    click.secho("\nMạng:", fg="green")
    click.echo(tabulate(list(network_config.items()), tablefmt="simple"))
    
    click.secho("\nGiao diện:", fg="green")
    click.echo(tabulate(list(ui_config.items()), tablefmt="simple"))
    
    click.secho("\nHiệu suất:", fg="green")
    click.echo(tabulate(list(performance_config.items()), tablefmt="simple"))
    
    if other_config:
        click.secho("\nKhác:", fg="green")
        click.echo(tabulate(list(other_config.items()), tablefmt="simple"))


@dashboard.command(help="Cập nhật cấu hình dashboard")
@click.argument("param")
@click.argument("value")
def set_config(param: str, value: str):
    """
    Cập nhật một tham số cấu hình dashboard.
    
    Args:
        param: Tên tham số
        value: Giá trị mới
    """
    manager = DashboardManager()
    
    # Chuyển đổi giá trị thành kiểu dữ liệu phù hợp
    if value.lower() == "true":
        value = True
    elif value.lower() == "false":
        value = False
    elif value.isdigit():
        value = int(value)
    elif value.replace(".", "", 1).isdigit():
        value = float(value)
    
    success, message = manager.update_config_param(param, value)
    
    if success:
        click.secho(message, fg="green")
    else:
        click.secho(message, fg="red")


@dashboard.command(help="Tạo URL truy cập public")
@click.option("--service", "-s", type=click.Choice(["ngrok", "localtunnel"]), default="ngrok", help="Dịch vụ tạo URL public")
@click.option("--region", "-r", default="auto", help="Khu vực cho ngrok (ví dụ: us, eu, ap)")
def create_public_url(service: str, region: str):
    """
    Tạo URL truy cập public cho dashboard.
    
    Args:
        service: Dịch vụ tạo URL public
        region: Khu vực cho ngrok
    """
    manager = DashboardManager()
    info = manager.get_dashboard_info()
    
    if not info["running"]:
        click.secho("❌ Dashboard không chạy", fg="red")
        return
    
    port = info["port"]
    
    if service == "ngrok":
        try:
            # Kiểm tra xem ngrok đã được cài đặt chưa
            try:
                subprocess.run(["ngrok", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                click.secho("Ngrok chưa được cài đặt. Vui lòng cài đặt ngrok:", fg="red")
                click.echo("  pip install pyngrok")
                return
            
            # Chạy ngrok
            cmd = ["ngrok", "http", f"{port}"]
            if region != "auto":
                cmd.extend(["--region", region])
            
            click.secho(f"Đang tạo URL public với ngrok cho port {port}...", fg="yellow")
            click.echo("Nhấn Ctrl+C để dừng.")
            
            # Chạy ngrok trong tiến trình con
            process = subprocess.Popen(cmd)
            
            # Đợi 2 giây
            time.sleep(2)
            
            # Lấy URL từ API của ngrok
            try:
                response = requests.get("http://localhost:4040/api/tunnels")
                data = response.json()
                public_url = data["tunnels"][0]["public_url"]
                click.secho(f"URL public: {public_url}", fg="green")
                
                # Đợi người dùng nhấn Ctrl+C
                try:
                    while True:
                        time.sleep(1)
                except KeyboardInterrupt:
                    process.terminate()
                    click.echo("Đã dừng ngrok.")
            except Exception as e:
                process.terminate()
                click.secho(f"Lỗi khi lấy URL ngrok: {str(e)}", fg="red")
        
        except Exception as e:
            click.secho(f"Lỗi khi tạo URL public với ngrok: {str(e)}", fg="red")
    
    elif service == "localtunnel":
        try:
            # Kiểm tra xem localtunnel đã được cài đặt chưa
            try:
                subprocess.run(["lt", "--version"], check=True, capture_output=True)
            except (subprocess.CalledProcessError, FileNotFoundError):
                click.secho("Localtunnel chưa được cài đặt. Vui lòng cài đặt localtunnel:", fg="red")
                click.echo("  npm install -g localtunnel")
                return
            
            # Chạy localtunnel
            cmd = ["lt", "--port", str(port)]
            
            click.secho(f"Đang tạo URL public với localtunnel cho port {port}...", fg="yellow")
            
            # Chạy localtunnel và lấy output
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            
            # Đọc output để lấy URL
            for line in process.stdout:
                if "your url is" in line.lower():
                    public_url = line.strip().split("is: ")[1]
                    click.secho(f"URL public: {public_url}", fg="green")
                    break
            
            # Đợi người dùng nhấn Ctrl+C
            click.echo("Nhấn Ctrl+C để dừng.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                process.terminate()
                click.echo("Đã dừng localtunnel.")
        
        except Exception as e:
            click.secho(f"Lỗi khi tạo URL public với localtunnel: {str(e)}", fg="red")


@dashboard.command(help="Mở dashboard trong trình duyệt")
def open():
    """Mở dashboard trong trình duyệt mặc định."""
    manager = DashboardManager()
    info = manager.get_dashboard_info()
    
    if not info["running"]:
        click.secho("❌ Dashboard không chạy", fg="red")
        return
    
    url = info["url"]
    
    # Mở URL trong trình duyệt mặc định
    click.secho(f"Đang mở {url}...", fg="green")
    
    # Xác định hệ điều hành
    if sys.platform == "win32":
        os.startfile(url)
    elif sys.platform == "darwin":
        subprocess.run(["open", url])
    else:
        subprocess.run(["xdg-open", url])

def setup_dashboard_parser(subparsers):
    """
    Thiết lập trình phân tích cú pháp cho các lệnh dashboard.
    
    Args:
        subparsers: Đối tượng subparsers từ argparse
    """
    dashboard_parser = subparsers.add_parser(
        'dashboard', 
        help='Quản lý dashboard giao dịch'
    )
    
    # Tạo các subcommand cho dashboard
    dashboard_subparsers = dashboard_parser.add_subparsers(
        dest='dashboard_command',
        help='Lệnh dashboard'
    )
    
    # Lệnh start
    start_parser = dashboard_subparsers.add_parser(
        'start', 
        help='Khởi động dashboard'
    )
    start_parser.add_argument(
        '--port', '-p', 
        type=int, 
        help='Cổng để chạy dashboard'
    )
    start_parser.add_argument(
        '--host', '-H', 
        default='localhost', 
        help='Địa chỉ host'
    )
    start_parser.add_argument(
        '--theme', '-t', 
        choices=['light', 'dark'], 
        default='light', 
        help='Chủ đề giao diện'
    )
    start_parser.add_argument(
        '--debug', 
        action='store_true', 
        help='Bật chế độ debug'
    )
    
    # Lệnh stop
    stop_parser = dashboard_subparsers.add_parser(
        'stop', 
        help='Dừng dashboard đang chạy'
    )
    stop_parser.add_argument(
        '--force', '-f', 
        action='store_true', 
        help='Buộc dừng dashboard'
    )
    
    # Lệnh restart
    dashboard_subparsers.add_parser(
        'restart', 
        help='Khởi động lại dashboard'
    )
    
    # Lệnh status
    status_parser = dashboard_subparsers.add_parser(
        'status', 
        help='Kiểm tra trạng thái dashboard'
    )
    status_parser.add_argument(
        '--json', 
        dest='json_output',
        action='store_true', 
        help='Xuất kết quả dạng JSON'
    )
    
    # Lệnh config
    config_parser = dashboard_subparsers.add_parser(
        'config', 
        help='Hiển thị cấu hình dashboard'
    )
    config_parser.add_argument(
        '--json', 
        dest='json_output',
        action='store_true', 
        help='Xuất kết quả dạng JSON'
    )
    
    # Lệnh set-config
    set_config_parser = dashboard_subparsers.add_parser(
        'set-config', 
        help='Cập nhật cấu hình dashboard'
    )
    set_config_parser.add_argument(
        'param', 
        help='Tên tham số'
    )
    set_config_parser.add_argument(
        'value', 
        help='Giá trị mới'
    )
    
    # Lệnh create-public-url
    public_url_parser = dashboard_subparsers.add_parser(
        'create-public-url', 
        help='Tạo URL truy cập public'
    )
    public_url_parser.add_argument(
        '--service', '-s', 
        choices=['ngrok', 'localtunnel'], 
        default='ngrok', 
        help='Dịch vụ tạo URL public'
    )
    public_url_parser.add_argument(
        '--region', '-r', 
        default='auto', 
        help='Khu vực cho ngrok (ví dụ: us, eu, ap)'
    )
    
    # Lệnh open
    dashboard_subparsers.add_parser(
        'open', 
        help='Mở dashboard trong trình duyệt'
    )
    
    return dashboard_parser

def handle_dashboard_command(args):
    """
    Xử lý các lệnh dashboard.
    
    Args:
        args: Đối tượng ArgumentParser đã phân tích
        
    Returns:
        int: Mã thoát (0 nếu thành công)
    """
    # Khởi tạo DashboardManager
    manager = DashboardManager()
    
    # Xử lý lệnh
    if args.dashboard_command == 'start':
        success, message = manager.start_dashboard(
            port=args.port,
            host=args.host,
            theme=args.theme,
            debug=args.debug
        )
        if success:
            print(message)
            return 0
        else:
            print(f"Lỗi: {message}")
            return 1
    
    elif args.dashboard_command == 'stop':
        success, message = manager.stop_dashboard(force=args.force)
        if success:
            print(message)
            return 0
        else:
            print(f"Lỗi: {message}")
            return 1
    
    elif args.dashboard_command == 'restart':
        success, message = manager.restart_dashboard()
        if success:
            print(message)
            return 0
        else:
            print(f"Lỗi: {message}")
            return 1
    
    elif args.dashboard_command == 'status':
        info = manager.get_dashboard_info()
        
        if args.json_output:
            import json
            print(json.dumps(info, indent=2))
        else:
            if info["running"]:
                print("✅ Dashboard đang chạy")
                print(f"PID: {info['pid']}")
                print(f"URL: {info['url']}")
                print(f"Thời gian chạy: {info['uptime']:.1f} giây")
                print(f"Bộ nhớ sử dụng: {info['memory_usage']:.2f} MB")
            else:
                print("❌ Dashboard không chạy")
        
        return 0
    
    elif args.dashboard_command == 'config':
        config_data = manager.load_config()
        
        if args.json_output:
            import json
            print(json.dumps(config_data, indent=2, ensure_ascii=False))
        else:
            print("Cấu hình Dashboard:")
            for key, value in config_data.items():
                print(f"  {key}: {value}")
        
        return 0
    
    elif args.dashboard_command == 'set-config':
        # Chuyển đổi giá trị thành kiểu phù hợp
        value = args.value
        if value.lower() == "true":
            value = True
        elif value.lower() == "false":
            value = False
        elif value.isdigit():
            value = int(value)
        
        success, message = manager.update_config_param(args.param, value)
        
        if success:
            print(message)
            return 0
        else:
            print(f"Lỗi: {message}")
            return 1
    
    elif args.dashboard_command == 'create-public-url':
        # Kiểm tra trạng thái dashboard
        info = manager.get_dashboard_info()
        
        if not info["running"]:
            print("❌ Dashboard không chạy")
            return 1
        
        port = info["port"]
        service = args.service
        region = args.region
        
        # Thực hiện lệnh bên ngoài để tạo URL public
        import subprocess
        import time
        
        if service == "ngrok":
            try:
                # Kiểm tra ngrok đã cài đặt chưa
                subprocess.run(["ngrok", "--version"], check=True, capture_output=True)
                
                print(f"Đang tạo URL public với ngrok cho port {port}...")
                print("Nhấn Ctrl+C để dừng.")
                
                # Chạy ngrok
                cmd = ["ngrok", "http", f"{port}"]
                if region != "auto":
                    cmd.extend(["--region", region])
                
                subprocess.Popen(cmd)
                
                # Đợi để ngrok khởi động
                time.sleep(2)
                
                return 0
            except Exception as e:
                print(f"Lỗi khi tạo URL với ngrok: {str(e)}")
                return 1
        
        elif service == "localtunnel":
            try:
                # Kiểm tra localtunnel đã cài đặt chưa
                subprocess.run(["lt", "--version"], check=True, capture_output=True)
                
                print(f"Đang tạo URL public với localtunnel cho port {port}...")
                print("Nhấn Ctrl+C để dừng.")
                
                # Chạy localtunnel
                subprocess.Popen(["lt", "--port", str(port)])
                
                # Đợi để localtunnel khởi động
                time.sleep(2)
                
                return 0
            except Exception as e:
                print(f"Lỗi khi tạo URL với localtunnel: {str(e)}")
                return 1
    
    elif args.dashboard_command == 'open':
        # Kiểm tra trạng thái dashboard
        info = manager.get_dashboard_info()
        
        if not info["running"]:
            print("❌ Dashboard không chạy")
            return 1
        
        # Mở trong trình duyệt
        url = info["url"]
        print(f"Đang mở {url}...")
        
        import webbrowser
        webbrowser.open(url)
        
        return 0
    
    else:
        print(f"Lệnh không hỗ trợ: {args.dashboard_command}")
        return 1

if __name__ == "__main__":
    dashboard()