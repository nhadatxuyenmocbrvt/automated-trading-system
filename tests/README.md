# Hướng dẫn chạy Unit Tests

Module này chứa tất cả các bài kiểm thử (unit tests) cho dự án Automated Trading System.

## Cấu trúc thư mục

```
tests/
├── __init__.py
├── __main__.py
├── README.md
├── technical_indicators/
│   ├── __init__.py
│   ├── test_data.py
│   ├── test_momentum_indicators.py
│   ├── test_support_resistance.py
│   ├── test_trend_indicators.py
│   ├── test_utils.py
│   ├── test_volatility_indicators.py
│   └── test_volume_indicators.py
└── ... (các module test khác)
```

## Cách chạy tests

### Chạy tất cả các tests

```bash
python -m tests
```

Lệnh này sẽ chạy tất cả các bài test trong dự án.

### Chạy một module test cụ thể

```bash
python -m tests.technical_indicators.test_trend_indicators
```

Lệnh này sẽ chạy tất cả các test trong module test_trend_indicators.py.

### Chạy một test case cụ thể

```bash
python -m unittest tests.technical_indicators.test_trend_indicators.TestTrendIndicators.test_simple_moving_average
```

Lệnh này sẽ chạy test case test_simple_moving_average trong TestTrendIndicators.

## Log files

Tất cả các logs khi chạy tests sẽ được lưu trong thư mục `logs/test_logs/`.

File log sẽ có tên theo định dạng `technical_indicators_tests_YYYYMMDD.log`, trong đó YYYYMMDD là ngày thực hiện tests.

## Tạo dữ liệu test

Module `test_data.py` trong mỗi thư mục test cung cấp các hàm để tạo dữ liệu test ngẫu nhiên với các đặc tính khác nhau:

- `generate_sample_price_data()`: Tạo dữ liệu giá OHLCV ngẫu nhiên thông thường
- `generate_trending_price_data()`: Tạo dữ liệu giá với xu hướng rõ ràng 
- `generate_volatile_price_data()`: Tạo dữ liệu giá với độ biến động cao