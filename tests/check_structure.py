"""
Script to check project structure and locate test files.
"""

import os
import sys

def print_directory_structure(start_path, indent=''):
    """Print directory structure starting from start_path."""
    print(f"{indent}+ {os.path.basename(start_path)}/")
    indent += '  '
    
    try:
        items = os.listdir(start_path)
    except PermissionError:
        print(f"{indent}Permission denied")
        return
    
    for item in sorted(items):
        item_path = os.path.join(start_path, item)
        if os.path.isdir(item_path):
            print_directory_structure(item_path, indent)
        else:
            print(f"{indent}- {item}")

def find_test_files(start_path):
    """Find test files recursively starting from start_path."""
    test_files = []
    
    for root, _, files in os.walk(start_path):
        for file in files:
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(root, file))
    
    return test_files

if __name__ == "__main__":
    # Get project root directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    print(f"Current directory: {current_dir}")
    print(f"Project root: {project_root}")
    
    # Print Python path
    print("\nPython path:")
    for path in sys.path:
        print(f"  {path}")
    
    # Print directory structure
    print("\nTests directory structure:")
    print_directory_structure(current_dir)
    
    # Find test files
    print("\nFound test files:")
    test_files = find_test_files(current_dir)
    for file in test_files:
        print(f"  {file}")
    
    # Check if test files exist
    for test_name in ['test_generic_connector.py', 'test_binance_connector.py', 'test_bybit_connector.py']:
        test_path = os.path.join(current_dir, test_name)
        exists = os.path.exists(test_path)
        print(f"\nCheck {test_name}: {'EXISTS' if exists else 'NOT FOUND'}")
        if not exists:
            # Try to find similar files
            print(f"Looking for similar files to {test_name}:")
            similar_files = [f for f in os.listdir(current_dir) if f.endswith('.py') and 'connector' in f.lower()]
            for similar in similar_files:
                print(f"  Found: {similar}")