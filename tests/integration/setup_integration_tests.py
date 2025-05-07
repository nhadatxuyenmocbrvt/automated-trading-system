import os
import sys
import logging
from pathlib import Path

# Thêm thư mục gốc vào sys.path để import các module
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Thiết lập logging cho kiểm thử
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"{project_root}/logs/integration_tests.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("integration_tests")

# Thiết lập môi trường kiểm thử
os.environ["ENV"] = "test"
os.environ["USE_TEST_DATA"] = "True"
os.environ["DISABLE_REAL_API"] = "True"