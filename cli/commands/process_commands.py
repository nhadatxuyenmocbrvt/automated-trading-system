"""
Module xử lý các lệnh xử lý dữ liệu.
"""

async def handle_process(system, args):
    """
    Xử lý lệnh process.
    
    Args:
        system: Hệ thống giao dịch tự động
        args: Tham số dòng lệnh
    """
    if not args.process_command:
        # Hiển thị help cho lệnh process
        from cli.parser import create_parser
        parser = create_parser()
        process_parser = [p for p in parser._subparsers._actions 
                        if hasattr(p, 'choices') and 'process' in p.choices][0].choices['process']
        process_parser.print_help()
        return
    
    # Chuyển đổi Namespace thành dict
    kwargs = vars(args)
    # Loại bỏ các tham số không cần thiết
    kwargs.pop('command', None)
    command = kwargs.pop('process_command', None)
    await system.process_data(command, **kwargs)