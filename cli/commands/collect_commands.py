"""
Module xử lý lệnh thu thập dữ liệu.
"""

async def handle_collect(system, args):
    """
    Xử lý lệnh collect.
    
    Args:
        system: Hệ thống giao dịch tự động
        args: Tham số dòng lệnh
    """
    await system.collect_historical_data(
        exchange_id=args.exchange,
        symbols=args.symbols,
        timeframes=args.timeframes,
        days_back=args.days,
        is_futures=args.futures
    )