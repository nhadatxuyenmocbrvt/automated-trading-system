import pandas as pd
import numpy as np
import os
import sys
import glob
import argparse
from pathlib import Path

def convert_timestamp_to_datetime(file_path, output_file=None):
    """
    Chuyển đổi cột timestamp trong file CSV sang định dạng datetime64[ns]
    
    Args:
        file_path (str): Đường dẫn đến file CSV cần xử lý
        output_file (str, optional): Đường dẫn file đầu ra. Nếu None, sẽ ghi đè file gốc.
    
    Returns:
        pd.DataFrame: DataFrame với cột timestamp đã được chuyển đổi
    """
    # Nếu không chỉ định file đầu ra, sử dụng file gốc
    if output_file is None:
        output_file = file_path.replace('.csv', '_fixed.csv')
    
    # Đọc file CSV
    print(f"Đang đọc file {os.path.basename(file_path)}...")
    df = pd.read_csv(file_path)
    
    # Kiểm tra xem timestamp có phải đã là datetime không
    if pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        print("Cột timestamp đã ở định dạng datetime64[ns], không cần chuyển đổi.")
        is_converted = False
    else:
        # Thử chuyển đổi trực tiếp nếu là chuỗi định dạng datetime
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("→ Đã chuyển đổi cột timestamp sang datetime64[ns] thành công.")
            is_converted = True
        except Exception as e:
            print(f"Không thể chuyển đổi trực tiếp: {str(e)}")
            
            # Nếu là số, kiểm tra xem là timestamp giây hay mili giây
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                max_value = df['timestamp'].max()
                
                if max_value > 1e12:  # Nếu là mili giây (13 chữ số)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                    print("→ Đã chuyển đổi timestamp từ mili giây sang datetime64[ns]")
                    is_converted = True
                else:  # Nếu là giây (10 chữ số)
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                    print("→ Đã chuyển đổi timestamp từ giây sang datetime64[ns]")
                    is_converted = True
            else:
                # Thử một số định dạng phổ biến khác
                formats = ['%Y-%m-%d', '%Y/%m/%d', '%d-%m-%Y', '%d/%m/%Y']
                for fmt in formats:
                    try:
                        df['timestamp'] = pd.to_datetime(df['timestamp'], format=fmt)
                        print(f"→ Đã chuyển đổi timestamp với định dạng {fmt}")
                        is_converted = True
                        break
                    except:
                        continue
                else:
                    print("Không thể xác định định dạng của cột timestamp.")
                    is_converted = False
    
    # Lưu lại file CSV với timestamp đã được chuyển đổi
    df.to_csv(output_file, index=False)
    print(f"➔ Đã lưu file kết quả: {output_file}")
    
    return df, is_converted

def generate_chart(df, output_path, pair_name="BTC/USDT"):
    """
    Tạo biểu đồ tâm lý thị trường.
    
    Args:
        df: DataFrame với dữ liệu tâm lý
        output_path: Đường dẫn lưu biểu đồ
        pair_name: Tên cặp giao dịch
    """
    try:
        import matplotlib.pyplot as plt
        
        # Đảm bảo thư mục đầu ra tồn tại
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        plt.figure(figsize=(12, 6))
        plt.plot(df['timestamp'], df['sentiment_score'], 'b-', linewidth=1.5)
        plt.fill_between(df['timestamp'], df['sentiment_score'], 0, 
                        where=(df['sentiment_score'] >= 0), color='green', alpha=0.3)
        plt.fill_between(df['timestamp'], df['sentiment_score'], 0, 
                        where=(df['sentiment_score'] < 0), color='red', alpha=0.3)
        
        plt.axhline(y=0, color='k', linestyle='-', alpha=0.2)
        plt.axhline(y=0.6, color='g', linestyle='--', alpha=0.2)
        plt.axhline(y=-0.6, color='r', linestyle='--', alpha=0.2)
        
        plt.title(f'Diễn biến chỉ số tâm lý thị trường {pair_name}')
        plt.ylabel('Điểm tâm lý')
        plt.xlabel('Thời gian')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path)
        print(f"➔ Đã lưu biểu đồ tâm lý thị trường: {output_path}")
        
    except ImportError:
        print("\n⚠️ Không thể vẽ biểu đồ vì thiếu thư viện matplotlib. Để vẽ biểu đồ, hãy cài đặt matplotlib bằng lệnh:")
        print("pip install matplotlib")

def extract_pair_name(file_path):
    """
    Trích xuất tên cặp giao dịch từ tên file.
    
    Args:
        file_path: Đường dẫn đến file
        
    Returns:
        Tên cặp giao dịch (vd: "BTC/USDT")
    """
    base_name = os.path.basename(file_path)
    # Tìm phần tên trước timestamp (thường là BTC_USDT)
    pair_with_underscore = base_name.split('_sentiment_')[0]
    # Chuyển định dạng về BTC/USDT
    if '_' in pair_with_underscore:
        parts = pair_with_underscore.split('_')
        pair_name = f"{parts[0]}/{parts[1]}"
        return pair_name
    return "Unknown/Unknown"  # Trường hợp không xác định được

def process_file(file_path, chart_dir=None):
    """
    Xử lý một file CSV.
    
    Args:
        file_path: Đường dẫn đến file
        chart_dir: Thư mục lưu biểu đồ (tùy chọn)
        
    Returns:
        True nếu xử lý thành công, False nếu có lỗi
    """
    try:
        # Xác định file đầu ra
        output_dir = os.path.dirname(file_path)
        output_file = os.path.join(output_dir, f"FIX_{os.path.basename(file_path)}")
        
        # Xử lý chuyển đổi timestamp
        df_converted, is_converted = convert_timestamp_to_datetime(file_path, output_file)
        
        if is_converted:
            # Trích xuất tên cặp giao dịch
            pair_name = extract_pair_name(file_path)
            
            # Tạo tên file biểu đồ (đảm bảo không chứa ký tự đặc biệt về đường dẫn)
            pair_for_filename = pair_name.replace('/', '_')
            
            # Xác định thư mục lưu biểu đồ
            if chart_dir is None:
                chart_dir = os.path.join(output_dir, "charts")
            
            # Tạo đường dẫn lưu biểu đồ
            chart_path = os.path.join(chart_dir, f"{pair_for_filename}_sentiment_chart.png")
            
            # Tạo biểu đồ
            generate_chart(df_converted, chart_path, pair_name)
            
        return True
    except Exception as e:
        print(f"❌ Lỗi khi xử lý file {file_path}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Hàm chính xử lý tham số dòng lệnh và điều phối xử lý các file."""
    parser = argparse.ArgumentParser(description='Chuyển đổi cột timestamp sang định dạng datetime64[ns]')
    parser.add_argument('--file', type=str, help='Đường dẫn đến file CSV cần xử lý')
    parser.add_argument('--dir', type=str, help='Thư mục chứa các file CSV cần xử lý')
    parser.add_argument('--pattern', type=str, default='*_sentiment_*.csv', 
                       help='Mẫu tên file cần xử lý (mặc định: *_sentiment_*.csv)')
    parser.add_argument('--chart-dir', type=str, help='Thư mục lưu biểu đồ (tùy chọn)')
    parser.add_argument('--all', action='store_true', help='Xử lý tất cả file trong thư mục data/sentiment')
    
    args = parser.parse_args()
    
    # Nếu không có tham số, hiển thị trợ giúp
    if len(sys.argv) == 1:
        parser.print_help()
        return 1
    
    files_to_process = []
    
    # Xác định danh sách file cần xử lý
    if args.all:
        # Tìm thư mục data/sentiment
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sentiment_dir = os.path.join(base_dir, "data", "sentiment")
        
        if os.path.exists(sentiment_dir):
            # Tìm tất cả file CSV trong thư mục sentiment và các thư mục con
            files_to_process = glob.glob(os.path.join(sentiment_dir, "**", "*_sentiment_*.csv"), recursive=True)
            # Loại bỏ các file đã được fix
            files_to_process = [f for f in files_to_process if "FIX_" not in os.path.basename(f)]
        else:
            print(f"❌ Không tìm thấy thư mục {sentiment_dir}")
            return 1
    elif args.file:
        if os.path.exists(args.file):
            files_to_process = [args.file]
        else:
            print(f"❌ Không tìm thấy file {args.file}")
            return 1
    elif args.dir:
        if os.path.exists(args.dir):
            files_to_process = glob.glob(os.path.join(args.dir, args.pattern))
            # Loại bỏ các file đã được fix
            files_to_process = [f for f in files_to_process if "FIX_" not in os.path.basename(f)]
        else:
            print(f"❌ Không tìm thấy thư mục {args.dir}")
            return 1
    
    # Xử lý từng file
    if files_to_process:
        print(f"Tìm thấy {len(files_to_process)} file cần xử lý.")
        
        # Tạo thư mục lưu biểu đồ
        if args.chart_dir:
            chart_dir = args.chart_dir
            os.makedirs(chart_dir, exist_ok=True)
        else:
            # Mặc định: thư mục charts trong thư mục chứa file đầu tiên
            chart_dir = os.path.join(os.path.dirname(files_to_process[0]), "charts")
            os.makedirs(chart_dir, exist_ok=True)
        
        # Xử lý từng file
        success_count = 0
        for file_path in files_to_process:
            if process_file(file_path, chart_dir):
                success_count += 1
                
        print(f"\n✓ Đã xử lý thành công {success_count}/{len(files_to_process)} file.")
        if success_count < len(files_to_process):
            print(f"⚠️ {len(files_to_process) - success_count} file gặp lỗi khi xử lý.")
            
        return 0
    else:
        print("❌ Không tìm thấy file nào để xử lý.")
        return 1

if __name__ == "__main__":
    sys.exit(main())