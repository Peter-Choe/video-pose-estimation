from .base import BaseConfig
from pathlib import Path

class LocalConfig(BaseConfig):
    DEBUG: bool = True
    ENV: str = "local"  # 에러 방지용으로 추가
    SHOW_GUI: bool = False  # default fallback if not in env file

    class Config:
        env_file = ".env.local"
