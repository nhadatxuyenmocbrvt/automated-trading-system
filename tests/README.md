# Tests for Automated Trading System

## Tổng quan
Thư mục này chứa các unit tests để kiểm tra tính đúng đắn và hiệu suất của hệ thống giao dịch tự động. Các bài kiểm tra được tổ chức theo module và sử dụng pytest làm framework chính.

## Cấu trúc thư mục
```
tests/
├── __init__.py                    # File init cho package tests
├── README.md                      # Tài liệu này
├── conftest.py                    # Các fixture chung cho pytest
├── fixtures/                      # Dữ liệu mẫu cho tests
│   └── mock_exchange_data.json    # Dữ liệu mock cho API sàn giao dịch
├── test_generic_connector.py      # Tests cho GenericExchangeConnector
├── test_binance_connector.py      # Tests cho BinanceConnector
└── test_bybit_connector.py        # Tests cho BybitConnector
```

## Chạy tests
Để chạy toàn bộ unit tests:
```bash
pytest tests/
```

Để chạy test cho một connector cụ thể:
```bash
pytest tests/test_binance_connector.py
```

Để chạy test với verbose và dừng lại ngay khi có test đầu tiên fail:
```bash
pytest -xvs tests/
```

## Các phụ thuộc cho testing
- pytest
- pytest-asyncio (cho async tests)
- unittest.mock (cho mocking)

## Logs và phân tích lỗi
Tất cả logs của tests được ghi vào file `tests/test_exchange_connectors.log` để phân tích nếu có lỗi. Cấu hình logging được thiết lập trong `conftest.py`.

## Mock data
Dữ liệu mock của các API sàn giao dịch được lưu trong file `fixtures/mock_exchange_data.json`. Dữ liệu này được tự động tạo nếu chưa tồn tại khi bạn chạy tests.

## Fixtures
Các fixtures phổ biến cho tất cả các tests được định nghĩa trong `conftest.py`:
- `mock_system_config`: Mock cho SystemConfig
- `mock_secret_manager`: Mock cho SecretManager
- `mock_exchange_data`: Data mẫu cho các API call
- `mock_ccxt_exchange`: Mock cho ccxt exchange
- `mock_ccxt_async_exchange`: Mock cho ccxt.async_support exchange
- `mock_websocket`: Mock cho websocket connection
- `patch_ccxt_exchanges`: Patch ccxt và ccxt.async_support để sử dụng mock
- `patch_websockets`: Patch websockets.connect để sử dụng mock
- `event_loop`: Event loop cho async tests

## Viết thêm tests
Khi thêm chức năng mới, hãy tuân theo các nguyên tắc sau:
1. Tạo test case mới trong file test tương ứng hoặc tạo file test mới cho module mới.
2. Sử dụng mock để cô lập các thành phần bên ngoài (API calls, databases, etc.)
3. Mỗi test method nên tập trung vào một chức năng duy nhất.
4. Sử dụng logging để ghi lại thông tin hữu ích cho việc debug.
5. Đảm bảo các test độc lập với nhau (không phụ thuộc vào trạng thái từ các test khác).

## Xử lý lỗi
Nếu tests thất bại, kiểm tra log file để biết thêm chi tiết:
```bash
cat tests/test_exchange_connectors.log
```

## Bảo trì
Tests nên được cập nhật khi có thay đổi trong cấu trúc hoặc hành vi của các connector. Đảm bảo kiểm tra tất cả các test đều pass trước khi commit code mới.