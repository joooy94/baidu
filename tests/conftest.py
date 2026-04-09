"""测试配置 — 将 src/ 目录加入 Python 搜索路径"""
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
