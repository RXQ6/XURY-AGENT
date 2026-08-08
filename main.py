"""根入口 shim：兼容 `python main.py "主题"` 的调用方式。"""
from src.main import main

if __name__ == "__main__":
    main()
