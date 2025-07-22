import streamlit as st
import os

log_dir = "/mnt/data/trade_logs"

try:
    os.makedirs(log_dir, exist_ok=True)
    st.write(f"Directory '{log_dir}' created or already exists.")
except Exception as e:
    st.error(f"Error creating directory: {e}")
