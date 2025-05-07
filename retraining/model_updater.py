"""
Quản lý cập nhật mô hình.
File này định nghĩa lớp ModelUpdater để quản lý việc cập nhật mô hình, lưu trữ,
đăng ký phiên bản mới và triển khai mô hình trong hệ thống.
"""

import os
import time
import json
import shutil
import logging
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Any, Set
import numpy as np

# Import các module từ hệ thống
import sys
import os

# Thêm thư mục gốc vào sys.path để import module
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config.logging_config import get_logger
from config.system_config import get_system_config, MODEL_DIR
from models.agents.base_agent import BaseAgent

class ModelUpdater:
    """
    Lớp quản lý việc cập nhật, lưu trữ và triển khai mô hình.
    
    Cung cấp các tính năng để:
    1. Đăng ký phiên bản mới của mô hình
    2. Quản lý danh sách phiên bản
    3. Cập nhật phiên bản hiện tại
    4. Rollback về phiên bản trước đó
    5. Triển khai mô hình
    """
    
    def __init__(
        self,
        agent_type: str,
        model_version: str = "v1.0.0",
        output_dir: Optional[Union[str, Path]] = None,
        keep_old_versions: int = 3,
        max_versions: int = 10,
        logger: Optional[logging.Logger] = None
    ):
        """
        Khởi tạo ModelUpdater.
        
        Args:
            agent_type: Loại agent (dqn, ppo, a2c, etc.)
            model_version: Phiên bản hiện tại của mô hình
            output_dir: Thư mục chứa các phiên bản mô hình
            keep_old_versions: Số lượng phiên bản cũ giữ lại
            max_versions: Số lượng phiên bản tối đa được lưu
            logger: Logger tùy chỉnh
        """
        # Thiết lập logger
        self.logger = logger or get_logger("model_updater")
        
        # Thiết lập cấu hình hệ thống
        self.system_config = get_system_config()
        
        # Lưu thông tin cơ bản
        self.agent_type = agent_type
        self.current_version = model_version
        self.keep_old_versions = keep_old_versions
        self.max_versions = max_versions
        
        # Thiết lập thư mục đầu ra
        if output_dir is None:
            self.output_dir = Path(MODEL_DIR) / 'versions' / agent_type
        else:
            self.output_dir = Path(output_dir)
        
        # Tạo thư mục nếu chưa tồn tại
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Thư mục cho phiên bản hiện tại
        self.current_dir = self.output_dir / self.current_version
        
        # Thư mục cho phiên bản dự phòng
        self.backup_dir = self.output_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Thư mục đang triển khai
        self.deployment_dir = self.output_dir / "deployed"
        self.deployment_dir.mkdir(exist_ok=True)
        
        # Thư mục bị lỗi
        self.failed_dir = self.output_dir / "failed"
        self.failed_dir.mkdir(exist_ok=True)
        
        # Tải danh sách phiên bản
        self.versions = self._load_versions()
        
        # Kiểm tra và đăng ký phiên bản hiện tại nếu chưa có
        if self.current_version not in self.versions:
            self._register_current_version()
        
        self.logger.info(
            f"Đã khởi tạo ModelUpdater cho agent {agent_type}, "
            f"phiên bản hiện tại: {self.current_version}, "
            f"tổng số phiên bản: {len(self.versions)}"
        )
    
    def _load_versions(self) -> Dict[str, Dict[str, Any]]:
        """
        Tải danh sách phiên bản từ file.
        
        Returns:
            Dict chứa thông tin các phiên bản
        """
        versions_file = self.output_dir / "versions.json"
        
        if versions_file.exists():
            try:
                with open(versions_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Lỗi khi tải danh sách phiên bản: {str(e)}")
                return {}
        else:
            return {}
    
    def _save_versions(self) -> bool:
        """
        Lưu danh sách phiên bản vào file.
        
        Returns:
            True nếu lưu thành công, False nếu không
        """
        versions_file = self.output_dir / "versions.json"
        
        try:
            with open(versions_file, "w", encoding="utf-8") as f:
                json.dump(self.versions, f, indent=4, ensure_ascii=False)
            return True
        except Exception as e:
            self.logger.error(f"Lỗi khi lưu danh sách phiên bản: {str(e)}")
            return False
    
    def _register_current_version(self) -> bool:
        """
        Đăng ký phiên bản hiện tại nếu chưa tồn tại.
        
        Returns:
            True nếu đăng ký thành công, False nếu không
        """
        # Nếu phiên bản này đã tồn tại, không làm gì
        if self.current_version in self.versions:
            return True
        
        # Tạo thông tin phiên bản
        version_info = {
            "version": self.current_version,
            "agent_type": self.agent_type,
            "created_at": datetime.now().isoformat(),
            "status": "active",
            "is_current": True,
            "model_path": None,
            "metadata": {
                "registered_by": "system",
                "registration_type": "initial"
            }
        }
        
        # Đăng ký phiên bản
        self.versions[self.current_version] = version_info
        
        # Lưu danh sách phiên bản
        if self._save_versions():
            self.logger.info(f"Đã đăng ký phiên bản hiện tại: {self.current_version}")
            return True
        else:
            return False
    
    def register_new_version(
        self,
        agent: Optional[BaseAgent] = None,
        version: Optional[str] = None,
        model_path: Optional[Union[str, Path]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Đăng ký phiên bản mới của mô hình.
        
        Args:
            agent: Agent cần đăng ký
            version: Phiên bản mới (tạo tự động nếu không cung cấp)
            model_path: Đường dẫn đến file mô hình
            metadata: Metadata bổ sung
            
        Returns:
            Dict kết quả đăng ký
        """
        try:
            # Tạo ID phiên bản nếu chưa có
            if version is None:
                version = self._generate_new_version()
            
            # Kiểm tra xem phiên bản đã tồn tại chưa
            if version in self.versions:
                self.logger.warning(f"Phiên bản {version} đã tồn tại")
                return {
                    "success": False,
                    "message": f"Phiên bản {version} đã tồn tại",
                    "version": version
                }
            
            # Kiểm tra tính hợp lệ của model_path
            if model_path is not None:
                model_path = Path(model_path)
                if not model_path.exists():
                    self.logger.error(f"File mô hình không tồn tại: {model_path}")
                    return {
                        "success": False,
                        "message": f"File mô hình không tồn tại: {model_path}",
                        "version": version
                    }
            
            # Tạo thư mục cho phiên bản mới
            version_dir = self.output_dir / version
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Sao chép mô hình nếu có
            target_model_path = None
            if model_path is not None:
                # Tạo tên file mục tiêu
                target_filename = f"{self.agent_type}_{version}.h5"
                target_model_path = version_dir / target_filename
                
                # Sao chép file
                shutil.copy2(model_path, target_model_path)
                
                self.logger.info(f"Đã sao chép mô hình từ {model_path} đến {target_model_path}")
            elif agent is not None:
                # Lưu agent trực tiếp
                target_filename = f"{self.agent_type}_{version}.h5"
                target_model_path = version_dir / target_filename
                
                agent.save_model(path=target_model_path)
                
                self.logger.info(f"Đã lưu agent trực tiếp vào {target_model_path}")
            
            # Tính toán checksum cho file mô hình
            checksum = None
            if target_model_path is not None and target_model_path.exists():
                checksum = self._calculate_checksum(target_model_path)
            
            # Tạo thông tin phiên bản
            version_info = {
                "version": version,
                "agent_type": self.agent_type,
                "created_at": datetime.now().isoformat(),
                "status": "pending",  # Chưa triển khai
                "is_current": False,
                "model_path": str(target_model_path) if target_model_path else None,
                "checksum": checksum,
                "metadata": metadata or {}
            }
            
            # Lưu thông tin vào file metadata
            metadata_path = version_dir / "metadata.json"
            with open(metadata_path, "w", encoding="utf-8") as f:
                json.dump(version_info, f, indent=4, ensure_ascii=False)
            
            # Đăng ký phiên bản
            self.versions[version] = version_info
            
            # Giới hạn số lượng phiên bản
            self._cleanup_old_versions()
            
            # Lưu danh sách phiên bản
            if not self._save_versions():
                self.logger.error(f"Lỗi khi lưu danh sách phiên bản sau khi đăng ký {version}")
                
                # Xóa thư mục đã tạo nếu lưu không thành công
                try:
                    shutil.rmtree(version_dir)
                except Exception:
                    pass
                
                return {
                    "success": False,
                    "message": "Lỗi khi lưu danh sách phiên bản",
                    "version": version
                }
            
            self.logger.info(f"Đã đăng ký phiên bản mới: {version}")
            
            return {
                "success": True,
                "message": f"Đã đăng ký phiên bản mới: {version}",
                "version": version,
                "model_path": str(target_model_path) if target_model_path else None,
                "version_dir": str(version_dir)
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi đăng ký phiên bản mới: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi đăng ký phiên bản mới: {str(e)}",
                "version": version if version else "unknown"
            }
    
    def update_current_version(
        self,
        version: str,
        gradual_rollout: bool = False,
        validation_period: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Cập nhật phiên bản hiện tại.
        
        Args:
            version: Phiên bản cần cập nhật thành hiện tại
            gradual_rollout: Triển khai dần dần
            validation_period: Thời gian xác thực (giờ)
            
        Returns:
            Dict kết quả cập nhật
        """
        try:
            # Kiểm tra xem phiên bản có tồn tại không
            if version not in self.versions:
                self.logger.error(f"Phiên bản không tồn tại: {version}")
                return {
                    "success": False,
                    "message": f"Phiên bản không tồn tại: {version}",
                    "version": version
                }
            
            # Kiểm tra xem phiên bản này đã là hiện tại chưa
            if version == self.current_version:
                self.logger.info(f"Phiên bản {version} đã là phiên bản hiện tại")
                return {
                    "success": True,
                    "message": f"Phiên bản {version} đã là phiên bản hiện tại",
                    "version": version,
                    "no_change": True
                }
            
            # Tạo bản sao của phiên bản hiện tại
            old_version = self.current_version
            
            # Tạo thư mục dự phòng cho phiên bản cũ
            backup_version_dir = self.backup_dir / old_version
            backup_version_dir.mkdir(parents=True, exist_ok=True)
            
            # Sao chép thông tin phiên bản cũ vào thư mục dự phòng
            old_version_dir = self.output_dir / old_version
            if old_version_dir.exists():
                for item in old_version_dir.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, backup_version_dir)
            
            # Lưu lịch sử cập nhật
            update_history = {
                "timestamp": datetime.now().isoformat(),
                "from_version": old_version,
                "to_version": version,
                "reason": self.versions[version].get("metadata", {}).get("reason", "manual_update"),
                "gradual_rollout": gradual_rollout,
                "validation_period": validation_period
            }
            
            # Lưu lịch sử cập nhật
            history_dir = self.output_dir / "history"
            history_dir.mkdir(exist_ok=True)
            
            history_file = history_dir / f"update_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(update_history, f, indent=4, ensure_ascii=False)
            
            # Cập nhật trạng thái phiên bản cũ
            if old_version in self.versions:
                self.versions[old_version]["is_current"] = False
                self.versions[old_version]["status"] = "inactive"
            
            # Cập nhật trạng thái phiên bản mới
            if gradual_rollout:
                # Nếu triển khai dần dần, đặt trạng thái là "validating"
                self.versions[version]["status"] = "validating"
                
                # Lưu thời gian kết thúc xác thực
                if validation_period is not None:
                    validation_end_time = datetime.now() + timedelta(hours=validation_period)
                    self.versions[version]["metadata"]["validation_end_time"] = validation_end_time.isoformat()
            else:
                # Nếu triển khai ngay lập tức, đặt trạng thái là "active"
                self.versions[version]["status"] = "active"
            
            # Luôn đặt is_current = True cho phiên bản mới
            self.versions[version]["is_current"] = True
            self.versions[version]["deployed_at"] = datetime.now().isoformat()
            
            # Cập nhật biến thành viên current_version
            self.current_version = version
            
            # Lưu danh sách phiên bản
            if not self._save_versions():
                self.logger.error(f"Lỗi khi lưu danh sách phiên bản sau khi cập nhật {version}")
                
                # Khôi phục phiên bản cũ
                self.current_version = old_version
                
                if old_version in self.versions:
                    self.versions[old_version]["is_current"] = True
                    self.versions[old_version]["status"] = "active"
                
                if version in self.versions:
                    self.versions[version]["is_current"] = False
                
                # Lưu lại danh sách phiên bản sau khi khôi phục
                self._save_versions()
                
                return {
                    "success": False,
                    "message": "Lỗi khi lưu danh sách phiên bản, đã khôi phục phiên bản cũ",
                    "version": old_version
                }
            
            self.logger.info(f"Đã cập nhật phiên bản hiện tại từ {old_version} thành {version}")
            
            return {
                "success": True,
                "message": f"Đã cập nhật phiên bản hiện tại từ {old_version} thành {version}",
                "old_version": old_version,
                "new_version": version,
                "gradual_rollout": gradual_rollout,
                "validation_period": validation_period
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi cập nhật phiên bản hiện tại: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi cập nhật phiên bản hiện tại: {str(e)}",
                "version": version
            }
    
    def complete_validation(
        self,
        version: Optional[str] = None,
        success: bool = True
    ) -> Dict[str, Any]:
        """
        Hoàn thành quá trình xác thực cho triển khai dần dần.
        
        Args:
            version: Phiên bản cần hoàn thành xác thực (mặc định là phiên bản hiện tại)
            success: Xác thực thành công hay thất bại
            
        Returns:
            Dict kết quả xác thực
        """
        if version is None:
            version = self.current_version
        
        try:
            # Kiểm tra xem phiên bản có tồn tại không
            if version not in self.versions:
                self.logger.error(f"Phiên bản không tồn tại: {version}")
                return {
                    "success": False,
                    "message": f"Phiên bản không tồn tại: {version}",
                    "version": version
                }
            
            # Kiểm tra xem phiên bản có đang trong trạng thái xác thực không
            if self.versions[version]["status"] != "validating":
                self.logger.warning(f"Phiên bản {version} không trong trạng thái xác thực, hiện tại là: {self.versions[version]['status']}")
                return {
                    "success": False,
                    "message": f"Phiên bản {version} không trong trạng thái xác thực",
                    "version": version,
                    "current_status": self.versions[version]["status"]
                }
            
            # Cập nhật trạng thái phiên bản
            if success:
                # Xác thực thành công, đặt trạng thái là "active"
                self.versions[version]["status"] = "active"
                validation_result = {
                    "success": True,
                    "message": f"Xác thực thành công cho phiên bản {version}",
                    "version": version,
                    "status": "active"
                }
            else:
                # Xác thực thất bại, đặt trạng thái là "failed"
                self.versions[version]["status"] = "failed"
                
                # Tìm phiên bản trước đó để rollback
                previous_version = self._find_previous_stable_version()
                
                if previous_version is not None:
                    # Rollback về phiên bản trước đó
                    rollback_result = self.rollback(previous_version)
                    
                    validation_result = {
                        "success": False,
                        "message": f"Xác thực thất bại cho phiên bản {version}, đã rollback về phiên bản {previous_version}",
                        "version": version,
                        "status": "failed",
                        "rollback_to": previous_version,
                        "rollback_result": rollback_result
                    }
                else:
                    validation_result = {
                        "success": False,
                        "message": f"Xác thực thất bại cho phiên bản {version}, không tìm thấy phiên bản ổn định để rollback",
                        "version": version,
                        "status": "failed"
                    }
            
            # Lưu danh sách phiên bản
            if not self._save_versions():
                self.logger.error(f"Lỗi khi lưu danh sách phiên bản sau khi hoàn thành xác thực {version}")
                return {
                    "success": False,
                    "message": "Lỗi khi lưu danh sách phiên bản",
                    "version": version
                }
            
            self.logger.info(f"Đã hoàn thành xác thực cho phiên bản {version}, kết quả: {'thành công' if success else 'thất bại'}")
            
            return validation_result
            
        except Exception as e:
            self.logger.error(f"Lỗi khi hoàn thành xác thực: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi hoàn thành xác thực: {str(e)}",
                "version": version
            }
    
    def rollback(self, to_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rollback về phiên bản cũ.
        
        Args:
            to_version: Phiên bản cần rollback (mặc định là tìm phiên bản ổn định gần nhất)
            
        Returns:
            Dict kết quả rollback
        """
        try:
            current_version = self.current_version
            
            # Nếu không chỉ định phiên bản cụ thể, tìm phiên bản ổn định trước đó
            if to_version is None:
                to_version = self._find_previous_stable_version()
                
                if to_version is None:
                    self.logger.error("Không tìm thấy phiên bản ổn định trước đó để rollback")
                    return {
                        "success": False,
                        "message": "Không tìm thấy phiên bản ổn định trước đó để rollback",
                        "current_version": current_version
                    }
            
            # Kiểm tra xem phiên bản có tồn tại không
            if to_version not in self.versions:
                self.logger.error(f"Phiên bản không tồn tại: {to_version}")
                return {
                    "success": False,
                    "message": f"Phiên bản không tồn tại: {to_version}",
                    "current_version": current_version
                }
            
            # Kiểm tra xem phiên bản này đã là hiện tại chưa
            if to_version == current_version:
                self.logger.info(f"Phiên bản {to_version} đã là phiên bản hiện tại")
                return {
                    "success": True,
                    "message": f"Phiên bản {to_version} đã là phiên bản hiện tại",
                    "version": to_version,
                    "no_change": True
                }
            
            # Kiểm tra xem phiên bản cần rollback có mô hình không
            if not self._check_model_exists(to_version):
                self.logger.error(f"Mô hình không tồn tại cho phiên bản {to_version}")
                return {
                    "success": False,
                    "message": f"Mô hình không tồn tại cho phiên bản {to_version}",
                    "version": to_version
                }
            
            # Tạo thư mục cho phiên bản hiện tại bị rollback vào thư mục failed
            failed_version_dir = self.failed_dir / current_version
            failed_version_dir.mkdir(parents=True, exist_ok=True)
            
            # Sao chép thông tin phiên bản hiện tại vào thư mục failed
            current_version_dir = self.output_dir / current_version
            if current_version_dir.exists():
                for item in current_version_dir.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, failed_version_dir)
            
            # Lưu lịch sử rollback
            rollback_history = {
                "timestamp": datetime.now().isoformat(),
                "from_version": current_version,
                "to_version": to_version,
                "reason": "rollback",
                "status": "completed"
            }
            
            # Lưu lịch sử rollback
            history_dir = self.output_dir / "history"
            history_dir.mkdir(exist_ok=True)
            
            history_file = history_dir / f"rollback_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(rollback_history, f, indent=4, ensure_ascii=False)
            
            # Cập nhật trạng thái phiên bản cũ
            if current_version in self.versions:
                self.versions[current_version]["is_current"] = False
                self.versions[current_version]["status"] = "rolled_back"
            
            # Cập nhật trạng thái phiên bản mới
            self.versions[to_version]["is_current"] = True
            self.versions[to_version]["status"] = "active"
            self.versions[to_version]["deployed_at"] = datetime.now().isoformat()
            self.versions[to_version]["metadata"]["rollback_from"] = current_version
            self.versions[to_version]["metadata"]["rollback_time"] = datetime.now().isoformat()
            
            # Cập nhật biến thành viên current_version
            self.current_version = to_version
            
            # Lưu danh sách phiên bản
            if not self._save_versions():
                self.logger.error(f"Lỗi khi lưu danh sách phiên bản sau khi rollback về {to_version}")
                
                # Khôi phục phiên bản cũ
                self.current_version = current_version
                
                if current_version in self.versions:
                    self.versions[current_version]["is_current"] = True
                    self.versions[current_version]["status"] = "active"
                
                if to_version in self.versions:
                    self.versions[to_version]["is_current"] = False
                
                # Lưu lại danh sách phiên bản sau khi khôi phục
                self._save_versions()
                
                return {
                    "success": False,
                    "message": "Lỗi khi lưu danh sách phiên bản, đã khôi phục phiên bản cũ",
                    "version": current_version
                }
            
            self.logger.info(f"Đã rollback từ phiên bản {current_version} về phiên bản {to_version}")
            
            return {
                "success": True,
                "message": f"Đã rollback từ phiên bản {current_version} về phiên bản {to_version}",
                "from_version": current_version,
                "to_version": to_version,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi rollback: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi rollback: {str(e)}",
                "current_version": self.current_version
            }
    
    def load_model(
        self,
        version: Optional[str] = None,
        agent: Optional[BaseAgent] = None
    ) -> Dict[str, Any]:
        """
        Tải mô hình của một phiên bản cụ thể.
        
        Args:
            version: Phiên bản cần tải (mặc định là phiên bản hiện tại)
            agent: Agent để tải mô hình vào
            
        Returns:
            Dict kết quả tải mô hình
        """
        if version is None:
            version = self.current_version
            
        if agent is None:
            self.logger.warning("Không có agent để tải mô hình vào")
            return {
                "success": False,
                "message": "Không có agent để tải mô hình vào",
                "version": version
            }
        
        try:
            # Kiểm tra xem phiên bản có tồn tại không
            if version not in self.versions:
                self.logger.error(f"Phiên bản không tồn tại: {version}")
                return {
                    "success": False,
                    "message": f"Phiên bản không tồn tại: {version}",
                    "version": version
                }
            
            # Lấy đường dẫn mô hình
            model_path = self.versions[version].get("model_path")
            
            if model_path is None or not Path(model_path).exists():
                # Thử tìm mô hình trong thư mục phiên bản
                version_dir = self.output_dir / version
                if version_dir.exists():
                    model_files = list(version_dir.glob(f"{self.agent_type}_{version}*.h5"))
                    if model_files:
                        model_path = str(model_files[0])
            
            if model_path is None or not Path(model_path).exists():
                self.logger.error(f"Không tìm thấy mô hình cho phiên bản {version}")
                return {
                    "success": False,
                    "message": f"Không tìm thấy mô hình cho phiên bản {version}",
                    "version": version
                }
            
            # Tải mô hình vào agent
            success = agent.load_model(model_path)
            
            if success:
                self.logger.info(f"Đã tải mô hình phiên bản {version} thành công")
                return {
                    "success": True,
                    "message": f"Đã tải mô hình phiên bản {version} thành công",
                    "version": version,
                    "model_path": model_path
                }
            else:
                self.logger.error(f"Lỗi khi tải mô hình phiên bản {version}")
                return {
                    "success": False,
                    "message": f"Lỗi khi tải mô hình phiên bản {version}",
                    "version": version,
                    "model_path": model_path
                }
                
        except Exception as e:
            self.logger.error(f"Lỗi khi tải mô hình: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi tải mô hình: {str(e)}",
                "version": version
            }
    
    def get_version_info(self, version: Optional[str] = None) -> Dict[str, Any]:
        """
        Lấy thông tin chi tiết về một phiên bản.
        
        Args:
            version: Phiên bản cần lấy thông tin (mặc định là phiên bản hiện tại)
            
        Returns:
            Dict chứa thông tin phiên bản
        """
        if version is None:
            version = self.current_version
        
        # Kiểm tra xem phiên bản có tồn tại không
        if version not in self.versions:
            self.logger.warning(f"Phiên bản không tồn tại: {version}")
            return {
                "success": False,
                "message": f"Phiên bản không tồn tại: {version}",
                "version": version
            }
        
        # Lấy thông tin phiên bản
        version_info = self.versions[version].copy()
        
        # Kiểm tra xem mô hình có tồn tại không
        model_path = version_info.get("model_path")
        model_exists = model_path is not None and Path(model_path).exists()
        version_info["model_exists"] = model_exists
        
        # Kiểm tra xem thư mục phiên bản có tồn tại không
        version_dir = self.output_dir / version
        version_exists = version_dir.exists()
        version_info["directory_exists"] = version_exists
        
        # Kiểm tra xem file metadata có tồn tại không
        metadata_path = version_dir / "metadata.json"
        metadata_exists = metadata_path.exists()
        version_info["metadata_exists"] = metadata_exists
        
        # Tải metadata từ file nếu có
        if metadata_exists:
            try:
                with open(metadata_path, "r", encoding="utf-8") as f:
                    file_metadata = json.load(f)
                    
                    # Cập nhật metadata với thông tin từ file
                    if "metadata" in file_metadata:
                        version_info["file_metadata"] = file_metadata["metadata"]
            except Exception as e:
                self.logger.warning(f"Lỗi khi đọc file metadata: {str(e)}")
        
        return {
            "success": True,
            "version": version,
            "info": version_info
        }
    
    def get_all_versions(self) -> List[Dict[str, Any]]:
        """
        Lấy danh sách tất cả các phiên bản.
        
        Returns:
            List các thông tin phiên bản
        """
        result = []
        
        for version, info in self.versions.items():
            # Thêm thông tin cơ bản
            version_summary = {
                "version": version,
                "status": info.get("status", "unknown"),
                "is_current": info.get("is_current", False),
                "created_at": info.get("created_at"),
                "agent_type": info.get("agent_type", self.agent_type)
            }
            
            # Kiểm tra xem mô hình có tồn tại không
            model_path = info.get("model_path")
            if model_path:
                version_summary["model_exists"] = Path(model_path).exists()
            
            # Thêm thông tin bổ sung
            if "deployed_at" in info:
                version_summary["deployed_at"] = info["deployed_at"]
            
            result.append(version_summary)
        
        # Sắp xếp theo thời gian tạo (mới nhất lên đầu)
        result.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        
        return result
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """
        Lấy trạng thái triển khai hiện tại.
        
        Returns:
            Dict chứa thông tin trạng thái triển khai
        """
        result = {
            "current_version": self.current_version,
            "agent_type": self.agent_type,
            "versions_count": len(self.versions),
            "active_versions": sum(1 for info in self.versions.values() if info.get("status") == "active"),
            "pending_versions": sum(1 for info in self.versions.values() if info.get("status") == "pending"),
            "validating_versions": sum(1 for info in self.versions.values() if info.get("status") == "validating"),
            "failed_versions": sum(1 for info in self.versions.values() if info.get("status") == "failed"),
            "timestamp": datetime.now().isoformat()
        }
        
        # Thêm thông tin phiên bản hiện tại
        if self.current_version in self.versions:
            current_info = self.versions[self.current_version]
            
            result["current_version_info"] = {
                "status": current_info.get("status", "unknown"),
                "created_at": current_info.get("created_at"),
                "deployed_at": current_info.get("deployed_at"),
                "model_path": current_info.get("model_path")
            }
            
            # Kiểm tra xem mô hình hiện tại có tồn tại không
            model_path = current_info.get("model_path")
            if model_path:
                result["current_version_info"]["model_exists"] = Path(model_path).exists()
        
        # Thông tin lần cập nhật gần nhất
        history_dir = self.output_dir / "history"
        if history_dir.exists():
            update_files = list(history_dir.glob("update_*.json"))
            if update_files:
                # Sắp xếp theo thời gian tạo file (mới nhất lên đầu)
                update_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
                
                try:
                    with open(update_files[0], "r", encoding="utf-8") as f:
                        last_update = json.load(f)
                        result["last_update"] = last_update
                except Exception:
                    pass
        
        return result
    
    def delete_version(self, version: str, force: bool = False) -> Dict[str, Any]:
        """
        Xóa một phiên bản khỏi hệ thống.
        
        Args:
            version: Phiên bản cần xóa
            force: Xóa cưỡng bức ngay cả khi là phiên bản hiện tại
            
        Returns:
            Dict kết quả xóa
        """
        try:
            # Kiểm tra xem phiên bản có tồn tại không
            if version not in self.versions:
                self.logger.warning(f"Phiên bản không tồn tại: {version}")
                return {
                    "success": False,
                    "message": f"Phiên bản không tồn tại: {version}",
                    "version": version
                }
            
            # Kiểm tra xem có phải phiên bản hiện tại không
            if version == self.current_version and not force:
                self.logger.error(f"Không thể xóa phiên bản hiện tại: {version}. Sử dụng force=True nếu thực sự muốn xóa.")
                return {
                    "success": False,
                    "message": f"Không thể xóa phiên bản hiện tại: {version}. Sử dụng force=True nếu thực sự muốn xóa.",
                    "version": version
                }
            
            # Lưu thông tin phiên bản trước khi xóa
            version_info = self.versions[version].copy()
            
            # Di chuyển file vào thư mục bị xóa
            deleted_dir = self.output_dir / "deleted"
            deleted_dir.mkdir(exist_ok=True)
            
            version_dir = self.output_dir / version
            if version_dir.exists():
                deleted_version_dir = deleted_dir / f"{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                deleted_version_dir.mkdir(exist_ok=True)
                
                for item in version_dir.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, deleted_version_dir)
                
                # Xóa thư mục phiên bản
                shutil.rmtree(version_dir)
            
            # Ghi log xóa phiên bản
            delete_info = {
                "version": version,
                "deleted_at": datetime.now().isoformat(),
                "was_current": version == self.current_version,
                "forced": force,
                "version_info": version_info
            }
            
            # Lưu lịch sử xóa
            history_dir = self.output_dir / "history"
            history_dir.mkdir(exist_ok=True)
            
            history_file = history_dir / f"delete_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(delete_info, f, indent=4, ensure_ascii=False)
            
            # Xóa phiên bản khỏi danh sách
            del self.versions[version]
            
            # Nếu xóa phiên bản hiện tại, cần tìm phiên bản mới
            if version == self.current_version:
                new_current = self._find_previous_stable_version()
                
                if new_current is not None:
                    # Cập nhật phiên bản hiện tại
                    self.current_version = new_current
                    
                    # Cập nhật trạng thái phiên bản mới
                    self.versions[new_current]["is_current"] = True
                    self.versions[new_current]["status"] = "active"
                else:
                    # Không tìm thấy phiên bản ổn định, đặt là None
                    self.current_version = None
            
            # Lưu danh sách phiên bản
            if not self._save_versions():
                self.logger.error(f"Lỗi khi lưu danh sách phiên bản sau khi xóa {version}")
                return {
                    "success": False,
                    "message": "Lỗi khi lưu danh sách phiên bản",
                    "version": version
                }
            
            self.logger.info(f"Đã xóa phiên bản {version}")
            
            return {
                "success": True,
                "message": f"Đã xóa phiên bản {version}",
                "version": version,
                "was_current": delete_info["was_current"],
                "new_current": self.current_version
            }
            
        except Exception as e:
            self.logger.error(f"Lỗi khi xóa phiên bản: {str(e)}")
            return {
                "success": False,
                "message": f"Lỗi khi xóa phiên bản: {str(e)}",
                "version": version
            }
    
    def _generate_new_version(self) -> str:
        """
        Tạo phiên bản mới dựa trên phiên bản hiện tại.
        
        Returns:
            Chuỗi phiên bản mới
        """
        try:
            # Phân tích phiên bản hiện tại
            parts = self.current_version.split('.')
            
            if len(parts) >= 3:
                # Semantic versioning: major.minor.patch
                major = int(parts[0].strip('v'))
                minor = int(parts[1])
                patch = int(parts[2])
                
                # Tăng patch version
                patch += 1
                
                return f"v{major}.{minor}.{patch}"
            else:
                # Phiên bản đơn giản
                try:
                    # Thử phân tích phiên bản là "v1", "v2", ...
                    version_num = int(self.current_version.strip('v'))
                    return f"v{version_num + 1}"
                except ValueError:
                    # Nếu không thể phân tích, thêm timestamp
                    timestamp = datetime.now().strftime("%Y%m%d%H%M")
                    return f"{self.current_version}_{timestamp}"
                
        except Exception:
            # Nếu không thể phân tích, tạo phiên bản mới với timestamp
            timestamp = datetime.now().strftime("%Y%m%d%H%M")
            return f"v1.0.0_{timestamp}"
    
    def _cleanup_old_versions(self) -> None:
        """
        Dọn dẹp các phiên bản cũ để giới hạn số lượng phiên bản lưu trữ.
        """
        if len(self.versions) <= self.max_versions:
            return
        
        # Lấy danh sách phiên bản theo thời gian tạo
        version_items = [(v, info.get("created_at", "")) for v, info in self.versions.items()]
        
        # Sắp xếp theo thời gian tạo (cũ nhất lên đầu)
        version_items.sort(key=lambda x: x[1])
        
        # Số phiên bản cần xóa
        to_delete_count = len(self.versions) - self.max_versions
        
        # Danh sách phiên bản cần xóa
        versions_to_delete = []
        
        # Duyệt qua các phiên bản cũ nhất
        for version, _ in version_items:
            # Không xóa phiên bản hiện tại
            if version == self.current_version:
                continue
                
            # Không xóa phiên bản đang trong trạng thái active hoặc validating
            status = self.versions[version].get("status")
            if status in ["active", "validating"]:
                continue
            
            versions_to_delete.append(version)
            
            # Nếu đủ số lượng cần xóa thì dừng
            if len(versions_to_delete) >= to_delete_count:
                break
        
        # Xóa các phiên bản
        for version in versions_to_delete:
            # Lưu thông tin phiên bản trước khi xóa
            version_info = self.versions[version].copy()
            
            # Di chuyển file vào thư mục bị xóa
            deleted_dir = self.output_dir / "deleted"
            deleted_dir.mkdir(exist_ok=True)
            
            version_dir = self.output_dir / version
            if version_dir.exists():
                deleted_version_dir = deleted_dir / f"{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                deleted_version_dir.mkdir(exist_ok=True)
                
                for item in version_dir.glob("*"):
                    if item.is_file():
                        shutil.copy2(item, deleted_version_dir)
                
                # Xóa thư mục phiên bản
                shutil.rmtree(version_dir)
            
            # Ghi log xóa phiên bản
            delete_info = {
                "version": version,
                "deleted_at": datetime.now().isoformat(),
                "reason": "cleanup_old_versions",
                "version_info": version_info
            }
            
            # Lưu lịch sử xóa
            history_dir = self.output_dir / "history"
            history_dir.mkdir(exist_ok=True)
            
            history_file = history_dir / f"cleanup_{version}_{datetime.now().strftime('%Y%m%d%H%M%S')}.json"
            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(delete_info, f, indent=4, ensure_ascii=False)
            
            # Xóa phiên bản khỏi danh sách
            del self.versions[version]
            
            self.logger.info(f"Đã xóa phiên bản cũ {version} trong quá trình dọn dẹp")
    
    def _find_previous_stable_version(self) -> Optional[str]:
        """
        Tìm phiên bản ổn định trước đó.
        
        Returns:
            Phiên bản ổn định trước đó hoặc None nếu không tìm thấy
        """
        # Lấy danh sách phiên bản theo thời gian triển khai
        stable_versions = []
        
        for version, info in self.versions.items():
            # Bỏ qua phiên bản hiện tại
            if version == self.current_version:
                continue
            
            # Chỉ xét các phiên bản đã từng active
            status = info.get("status")
            if status == "active":
                # Nếu có thời gian triển khai, thêm vào danh sách
                if "deployed_at" in info:
                    stable_versions.append((version, info["deployed_at"]))
        
        if not stable_versions:
            return None
        
        # Sắp xếp theo thời gian triển khai (mới nhất lên đầu)
        stable_versions.sort(key=lambda x: x[1], reverse=True)
        
        # Trả về phiên bản ổn định mới nhất
        return stable_versions[0][0]
    
    def _check_model_exists(self, version: str) -> bool:
        """
        Kiểm tra xem mô hình của phiên bản có tồn tại không.
        
        Args:
            version: Phiên bản cần kiểm tra
            
        Returns:
            True nếu mô hình tồn tại, False nếu không
        """
        if version not in self.versions:
            return False
        
        # Kiểm tra model_path trong thông tin phiên bản
        model_path = self.versions[version].get("model_path")
        if model_path and Path(model_path).exists():
            return True
        
        # Tìm trong thư mục phiên bản
        version_dir = self.output_dir / version
        if version_dir.exists():
            model_files = list(version_dir.glob(f"{self.agent_type}_{version}*.h5"))
            if model_files:
                # Cập nhật model_path
                self.versions[version]["model_path"] = str(model_files[0])
                self._save_versions()
                return True
        
        return False
    
    def _calculate_checksum(self, file_path: Union[str, Path]) -> str:
        """
        Tính toán checksum của file.
        
        Args:
            file_path: Đường dẫn đến file
            
        Returns:
            Chuỗi checksum
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return ""
        
        try:
            hasher = hashlib.md5()
            with open(file_path, "rb") as f:
                for chunk in iter(lambda: f.read(4096), b""):
                    hasher.update(chunk)
            return hasher.hexdigest()
        except Exception as e:
            self.logger.error(f"Lỗi khi tính toán checksum: {str(e)}")
            return ""


def create_model_updater(
    agent_type: str,
    model_version: str = "v1.0.0",
    output_dir: Optional[Union[str, Path]] = None,
    keep_old_versions: int = 3,
    logger: Optional[logging.Logger] = None
) -> ModelUpdater:
    """
    Hàm tiện ích để tạo ModelUpdater.
    
    Args:
        agent_type: Loại agent (dqn, ppo, a2c, etc.)
        model_version: Phiên bản hiện tại của mô hình
        output_dir: Thư mục chứa các phiên bản mô hình
        keep_old_versions: Số lượng phiên bản cũ giữ lại
        logger: Logger tùy chỉnh
        
    Returns:
        ModelUpdater đã được cấu hình
    """
    return ModelUpdater(
        agent_type=agent_type,
        model_version=model_version,
        output_dir=output_dir,
        keep_old_versions=keep_old_versions,
        logger=logger
    )