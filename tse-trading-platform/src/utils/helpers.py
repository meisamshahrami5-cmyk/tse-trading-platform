"""
توابع کمکی و عمومی پروژه
"""
import os
import yaml
from datetime import datetime
from pathlib import Path

def load_config(config_path: str = "config/config.yaml") -> dict:
    """بارگذاری فایل تنظیمات"""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def ensure_directory(path: str):
    """اطمینان از وجود پوشه"""
    Path(path).mkdir(parents=True, exist_ok=True)

def get_timestamp() -> str:
    """دریافت timestamp فعلی"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def print_separator(title: str = "", char: str = "=", length: int = 60):
    """چاپ خط جداکننده"""
    if title:
        print(f"\n{char * length}")
        print(f"  {title}")
        print(f"{char * length}\n")
    else:
        print(char * length)