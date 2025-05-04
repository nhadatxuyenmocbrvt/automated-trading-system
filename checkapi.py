import ccxt

# Điền API key/secret tại đây
api_key = "gwIQpSlP7PomIpn6zF6l9YyiS3lvogviaE4ADQDQjCjTiXCoFp5VwGJmhsd8TSFe"
api_secret = "beM69btbIYoHAlF9PXuALn7xiosfqgJDih9j0BJMFIRHOQHmkaLDCEhgbr5PrePU"

binance_futures = ccxt.binanceusdm({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True,
    'options': {'defaultType': 'future'},
})

# Kiểm tra account balance
try:
    balance = binance_futures.fetch_balance()
    print("✅ API Binance Futures hoạt động. Số dư hiện tại:")
    print(balance['total'])
except Exception as e:
    print("❌ Lỗi kết nối với Binance Futures:")
    print(str(e))
