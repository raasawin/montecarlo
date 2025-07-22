import os

log_dir = "/mnt/data/trade_logs"

try:
    os.makedirs(log_dir, exist_ok=True)
    print(f"Directory '{log_dir}' created or already exists.")
except Exception as e:
    print(f"Error creating directory: {e}")
