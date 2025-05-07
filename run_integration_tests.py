import unittest
import sys
import argparse
import logging
from pathlib import Path

# Thiết lập logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("logs/test_runner.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("test_runner")

def run_tests(test_type="all"):
    """Chạy các bài kiểm thử dựa trên loại đã chọn"""
    logger.info(f"Running {test_type} tests")
    
    # Xác định thư mục test
    if test_type == "integration":
        test_dir = "tests/integration"
    elif test_type == "performance":
        test_dir = "tests/performance"
    elif test_type == "edge_cases":
        test_dir = "tests/edge_cases"
    else:
        test_dir = "tests"
    
    # Tìm và chạy các test
    loader = unittest.TestLoader()
    suite = loader.discover(test_dir)
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Trả về kết quả cho CI/CD
    return 0 if result.wasSuccessful() else 1

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run automated tests")
    parser.add_argument(
        "--type", 
        choices=["all", "integration", "performance", "edge_cases"],
        default="all",
        help="Type of tests to run"
    )
    
    args = parser.parse_args()
    exit_code = run_tests(args.type)
    sys.exit(exit_code)