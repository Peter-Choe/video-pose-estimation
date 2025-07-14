from .base import BaseConfig

class DockerConfig(BaseConfig):
    DEBUG: bool = False
    ENV: str = "docker"  # 에러 방지용으로 추가

    class Config:
        env_file = ".env.docker"
